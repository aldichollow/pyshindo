"""Causal approximation filters used for real-time seismic intensity.

The 2008 and 2012 designs are reconstructed from the equations published by
Kunugi and co-authors rather than copied from a coefficient table. The optional
low-rate design applies the generalized second-integrator equations disclosed
in JP7681907B2. At 100 Hz, gamma = 1/12 reduces those generalized sections to
the 2012 Boxer--Thaler sections.

References
----------
Kunugi, T. et al. (2008), *A Real-Time Processing of Seismic Intensity*,
Journal of the Seismological Society of Japan, 60, 243--252.
https://doi.org/10.4294/zisin.60.243

Kunugi, T. et al. (2013), *An Improved Approximating Filter for Real-Time
Calculation of Seismic Intensity*, Journal of the Seismological Society of
Japan, 65, 223--230. https://doi.org/10.4294/zisin.65.223

Japanese patents JP4229337B2, JP5946067B2, and JP7681907B2. See the repository's
``PATENTS.md`` before use or redistribution.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from scipy.signal import sosfreqz

from ..exceptions import UnstableFilterError
from ..models import FilterStage, FrequencyResponse, RecursiveFilterDesign
from ..units import FloatArray

_GAIN_STAGE_NAME = "gain"


def _gain_stage(gain: float) -> FilterStage:
    return FilterStage(_GAIN_STAGE_NAME, None, np.array([gain, 0.0, 0.0, 1.0, 0.0, 0.0]))


class RealtimeFilter(StrEnum):
    """Available time-domain approximation-filter selections."""

    AUTO = "auto"
    KUNUGI_2008 = "kunugi2008"
    KUNUGI_2012 = "kunugi2012"
    JP7681907_LOWRATE = "jp7681907-lowrate"

    @classmethod
    def parse(cls, value: str | RealtimeFilter) -> RealtimeFilter:
        """Return a canonical filter identifier."""
        if isinstance(value, cls):
            return value
        normalized = value.strip().lower().replace("_", "").replace("-", "")
        aliases = {
            "auto": cls.AUTO,
            "kunugi2008": cls.KUNUGI_2008,
            "2008": cls.KUNUGI_2008,
            "original": cls.KUNUGI_2008,
            "kunugi2012": cls.KUNUGI_2012,
            "2012": cls.KUNUGI_2012,
            "improved": cls.KUNUGI_2012,
            "jp7681907lowrate": cls.JP7681907_LOWRATE,
            "jp7681907": cls.JP7681907_LOWRATE,
            "lowrate": cls.JP7681907_LOWRATE,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            supported = ", ".join(item.value for item in cls)
            message = f"Unsupported real-time filter {value!r}; choose {supported}."
            raise ValueError(message) from exc


class LowRateGammaPolicy(StrEnum):
    """Source-described strategies for generalized integrator parameters."""

    PIECEWISE = "piecewise"
    CONSTANT_ACCURATE = "constant-accurate"
    CONSTANT_STABLE = "constant-stable"

    @classmethod
    def parse(cls, value: str | LowRateGammaPolicy) -> LowRateGammaPolicy:
        """Return a canonical low-rate gamma policy."""
        if isinstance(value, cls):
            return value
        normalized = value.strip().lower().replace("_", "-")
        aliases = {
            "piecewise": cls.PIECEWISE,
            "constant-accurate": cls.CONSTANT_ACCURATE,
            "constant-stable": cls.CONSTANT_STABLE,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(f"Unsupported gamma policy {value!r}; choose {supported}.") from exc

    @property
    def requires_standard_frequencies(self) -> bool:
        """Return whether the published table assumes 0.5, 12, 20, and 30 Hz."""
        return self is LowRateGammaPolicy.PIECEWISE


@dataclass(frozen=True, slots=True)
class Kunugi2008Parameters:
    """Parameters reported for the original real-time approximation filter."""

    f0_hz: float = 0.45
    f1_hz: float = 7.0
    f2_hz: float = 11.0
    damping: float = 0.9
    gain: float = 1.409


@dataclass(frozen=True, slots=True)
class Kunugi2012Parameters:
    """Parameters reported for the improved real-time approximation filter."""

    f0_hz: float = 0.45
    f1_hz: float = 7.0
    correction_hz: float = 0.5
    lowpass_1_hz: float = 12.0
    lowpass_2_hz: float = 20.0
    lowpass_3_hz: float = 30.0
    correction_numerator_damping: float = 1.0
    correction_denominator_damping: float = 0.75
    lowpass_1_damping: float = 0.9
    lowpass_2_damping: float = 0.6
    lowpass_3_damping: float = 0.6
    gain: float = 1.262


@dataclass(frozen=True, slots=True)
class LowRateGammaSet:
    """Gamma values for the generalized second-integrator approximation.

    A numerator and denominator gamma are retained separately for every
    second-order stage. :func:`published_lowrate_gamma_set` returns one of the
    source-described implementations in which each numerator value equals its
    corresponding denominator value. Advanced users may supply an independently
    validated set.
    """

    correction_numerator: float
    correction_denominator: float
    lowpass_1_numerator: float
    lowpass_1_denominator: float
    lowpass_2_numerator: float
    lowpass_2_denominator: float
    lowpass_3_numerator: float
    lowpass_3_denominator: float

    @classmethod
    def equal_values(
        cls,
        correction: float,
        lowpass_1: float,
        lowpass_2: float,
        lowpass_3: float,
    ) -> LowRateGammaSet:
        """Create a set with equal numerator and denominator values per stage."""
        return cls(
            correction_numerator=correction,
            correction_denominator=correction,
            lowpass_1_numerator=lowpass_1,
            lowpass_1_denominator=lowpass_1,
            lowpass_2_numerator=lowpass_2,
            lowpass_2_denominator=lowpass_2,
            lowpass_3_numerator=lowpass_3,
            lowpass_3_denominator=lowpass_3,
        )

    @property
    def denominator_values(self) -> tuple[float, float, float, float]:
        """Return denominator gamma values in correction/LP1/LP2/LP3 order."""
        return (
            self.correction_denominator,
            self.lowpass_1_denominator,
            self.lowpass_2_denominator,
            self.lowpass_3_denominator,
        )


def _validate_parameters(parameters: Kunugi2008Parameters | Kunugi2012Parameters) -> None:
    for name, value in asdict(parameters).items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        if value <= 0.0:
            raise ValueError(f"{name} must be greater than zero.")


def _validate_gamma_set(gammas: LowRateGammaSet) -> None:
    for name, value in asdict(gammas).items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        if value <= 0.0:
            raise ValueError(f"{name} must be greater than zero.")


def _piecewise_gammas(rate: float) -> tuple[float, float, float, float]:
    if rate >= 80.0:
        return (1.0 / 12.0,) * 4
    if rate >= 70.0:
        return (1.0 / 12.0, 1.0 / 12.0, 1.0 / 12.0, 1.0 / 8.0)
    if rate >= 60.0:
        return (1.0 / 12.0, 1.0 / 12.0, 1.0 / 12.0, 1.0 / 6.0)
    if rate >= 50.0:
        return (1.0 / 12.0, 1.0 / 12.0, 1.0 / 10.0, 1.0 / 4.0)
    if rate >= 40.0:
        return (1.0 / 12.0, 1.0 / 12.0, 1.0 / 6.0, 1.0 / 4.0)
    if rate >= 30.0:
        return (1.0 / 12.0, 1.0 / 10.0, 1.0 / 4.0, 1.0 / 4.0)
    if rate >= 5.0:
        return (1.0 / 12.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0)
    return (1.0 / 6.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0)


def published_lowrate_gamma_set(
    sampling_rate_hz: float,
    *,
    policy: str | LowRateGammaPolicy = LowRateGammaPolicy.PIECEWISE,
) -> LowRateGammaSet:
    """Return source-described gamma values for the generalized low-rate filter.

    ``piecewise`` reproduces the finest-grained selection table in
    JP7681907B2 for 0.5, 12, 20, and 30 Hz stages. ``constant-accurate`` uses
    gamma = 1/12, while ``constant-stable`` uses gamma = 1/4 -- the
    source-described choice that guarantees stability for arbitrary positive
    characteristic frequencies. The source describes rates down to 1 Hz; lower
    rates are rejected rather than extrapolated.

    Stability does not imply fidelity. When characteristic frequencies exceed
    Nyquist, the digital response may differ substantially from the 100 Hz
    reference and must be validated for the intended use.
    """
    rate = float(sampling_rate_hz)
    if not math.isfinite(rate) or rate < 1.0:
        raise ValueError("sampling_rate_hz must be finite and at least 1 Hz.")
    selected = LowRateGammaPolicy.parse(policy)
    if selected is LowRateGammaPolicy.PIECEWISE:
        values = _piecewise_gammas(rate)
    elif selected is LowRateGammaPolicy.CONSTANT_ACCURATE:
        values = (1.0 / 12.0,) * 4
    else:
        values = (1.0 / 4.0,) * 4
    return LowRateGammaSet.equal_values(*values)


def lowrate_stability_lower_bounds(
    sampling_rate_hz: float,
    parameters: Kunugi2012Parameters | None = None,
) -> tuple[float, float, float, float]:
    """Return denominator-gamma lower bounds from JP7681907B2 equations 21--24.

    Values are returned in correction, low-pass 1, low-pass 2, and low-pass 3
    order. A selected denominator gamma must be at least its corresponding bound
    for the generalized second-order section to satisfy the disclosed pole
    stability condition.
    """
    rate = float(sampling_rate_hz)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("sampling_rate_hz must be finite and greater than zero.")
    p = parameters or Kunugi2012Parameters()
    _validate_parameters(p)
    dt = 1.0 / rate
    frequencies = (
        p.correction_hz,
        p.lowpass_1_hz,
        p.lowpass_2_hz,
        p.lowpass_3_hz,
    )
    a, b, c, d = (0.25 - 1.0 / (2.0 * np.pi * frequency * dt) ** 2 for frequency in frequencies)
    return (a, b, c, d)


def lowrate_gamma_stability_margins(
    sampling_rate_hz: float,
    gammas: LowRateGammaSet,
    parameters: Kunugi2012Parameters | None = None,
) -> tuple[float, float, float, float]:
    """Return selected denominator gamma minus each disclosed stability bound."""
    _validate_gamma_set(gammas)
    lower = lowrate_stability_lower_bounds(sampling_rate_hz, parameters)
    a, b, c, d = (
        gamma - bound
        for gamma, bound in zip(gammas.denominator_values, lower, strict=True)
    )
    return (a, b, c, d)


def _normalize_section(
    alpha0: float,
    alpha1: float,
    alpha2: float,
    beta0: float,
    beta1: float,
    beta2: float,
) -> FloatArray:
    if not math.isfinite(alpha0) or alpha0 == 0.0:
        raise ValueError("A recursive filter section has an invalid alpha0 coefficient.")
    section = np.asarray(
        [
            beta0 / alpha0,
            beta1 / alpha0,
            beta2 / alpha0,
            1.0,
            alpha1 / alpha0,
            alpha2 / alpha0,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(section)):
        raise ValueError("A recursive filter section contains a non-finite coefficient.")
    return section


def _first_order_section(a: float, b: float, frequency_hz: float, dt: float) -> FloatArray:
    omega = 2.0 * np.pi * frequency_hz
    alpha0 = omega + 2.0 * b / dt
    alpha1 = omega - 2.0 * b / dt
    beta0 = omega * a + 2.0 / dt
    beta1 = omega * a - 2.0 / dt
    return _normalize_section(alpha0, alpha1, 0.0, beta0, beta1, 0.0)


def _boxer_thaler_lowpass(frequency_hz: float, damping: float, dt: float) -> FloatArray:
    omega = 2.0 * np.pi * frequency_hz
    inv_dt = 1.0 / dt
    alpha0 = 12.0 * inv_dt**2 + 12.0 * damping * omega * inv_dt + omega**2
    alpha1 = 10.0 * omega**2 - 24.0 * inv_dt**2
    alpha2 = 12.0 * inv_dt**2 - 12.0 * damping * omega * inv_dt + omega**2
    beta0 = omega**2
    beta1 = 10.0 * omega**2
    beta2 = omega**2
    return _normalize_section(alpha0, alpha1, alpha2, beta0, beta1, beta2)


def _combine_sections(first: FloatArray, second: FloatArray) -> FloatArray:
    numerator = np.convolve(first[:3], second[:3])
    denominator = np.convolve(first[3:], second[3:])
    # Both inputs are first-order sections represented with a trailing zero.
    return np.asarray(
        [
            numerator[0],
            numerator[1],
            numerator[2],
            denominator[0],
            denominator[1],
            denominator[2],
        ],
        dtype=np.float64,
    )


def _kunugi_2008_sos(sampling_rate_hz: float, parameters: Kunugi2008Parameters) -> FloatArray:
    _validate_parameters(parameters)
    dt = 1.0 / sampling_rate_hz
    sections = [
        _first_order_section(0.0, 1.0, parameters.f0_hz, dt),
        _first_order_section(1.0, 2.0, parameters.f1_hz, dt),
        _first_order_section(4.0, 8.0, parameters.f1_hz, dt),
        _first_order_section(0.25, 0.5, parameters.f1_hz, dt),
    ]
    paired_1 = _combine_sections(sections[0], sections[1])
    paired_2 = _combine_sections(sections[2], sections[3])
    lowpass = _boxer_thaler_lowpass(parameters.f2_hz, parameters.damping, dt)
    sos = np.vstack([paired_1, paired_2, lowpass])
    sos[0, :3] *= parameters.gain
    return sos


def _kunugi_2008_stages(
    sampling_rate_hz: float,
    parameters: Kunugi2008Parameters,
) -> list[FilterStage]:
    """Return the five named analog factors behind :func:`_kunugi_2008_sos`.

    Each stage is computed independently with the same per-factor transform
    used to build the combined design, so cascading all of them (plus the
    gain stage) reproduces the combined ``sos`` exactly.
    """
    dt = 1.0 / sampling_rate_hz
    f0, f1, f2 = parameters.f0_hz, parameters.f1_hz, parameters.f2_hz
    return [
        FilterStage("period effect", f0, _first_order_section(0.0, 1.0, f0, dt)),
        FilterStage("7 Hz factor 1", f1, _first_order_section(1.0, 2.0, f1, dt)),
        FilterStage("7 Hz factor 2", f1, _first_order_section(4.0, 8.0, f1, dt)),
        FilterStage("7 Hz factor 3", f1, _first_order_section(0.25, 0.5, f1, dt)),
        FilterStage("high-cut", f2, _boxer_thaler_lowpass(f2, parameters.damping, dt)),
        _gain_stage(parameters.gain),
    ]


def _paired_base_section(f0_hz: float, f1_hz: float, dt: float) -> FloatArray:
    omega1 = 2.0 * np.pi * f0_hz
    omega2 = 2.0 * np.pi * f1_hz
    inv_dt = 1.0 / dt
    alpha0 = 8.0 * inv_dt**2 + (4.0 * omega1 + 2.0 * omega2) * inv_dt + omega1 * omega2
    alpha1 = 2.0 * omega1 * omega2 - 16.0 * inv_dt**2
    alpha2 = 8.0 * inv_dt**2 - (4.0 * omega1 + 2.0 * omega2) * inv_dt + omega1 * omega2
    beta0 = 4.0 * inv_dt**2 + 2.0 * omega2 * inv_dt
    beta1 = -8.0 * inv_dt**2
    beta2 = 4.0 * inv_dt**2 - 2.0 * omega2 * inv_dt
    return _normalize_section(alpha0, alpha1, alpha2, beta0, beta1, beta2)


def _paired_half_order_section(frequency_hz: float, dt: float) -> FloatArray:
    omega = 2.0 * np.pi * frequency_hz
    inv_dt = 1.0 / dt
    alpha0 = 16.0 * inv_dt**2 + 17.0 * omega * inv_dt + omega**2
    alpha1 = 2.0 * omega**2 - 32.0 * inv_dt**2
    alpha2 = 16.0 * inv_dt**2 - 17.0 * omega * inv_dt + omega**2
    beta0 = 4.0 * inv_dt**2 + 8.5 * omega * inv_dt + omega**2
    beta1 = 2.0 * omega**2 - 8.0 * inv_dt**2
    beta2 = 4.0 * inv_dt**2 - 8.5 * omega * inv_dt + omega**2
    return _normalize_section(alpha0, alpha1, alpha2, beta0, beta1, beta2)


def _correction_section(
    frequency_hz: float,
    numerator_damping: float,
    denominator_damping: float,
    dt: float,
) -> FloatArray:
    omega = 2.0 * np.pi * frequency_hz
    inv_dt = 1.0 / dt
    alpha0 = 12.0 * inv_dt**2 + 12.0 * denominator_damping * omega * inv_dt + omega**2
    alpha1 = 10.0 * omega**2 - 24.0 * inv_dt**2
    alpha2 = 12.0 * inv_dt**2 - 12.0 * denominator_damping * omega * inv_dt + omega**2
    beta0 = 12.0 * inv_dt**2 + 12.0 * numerator_damping * omega * inv_dt + omega**2
    beta1 = 10.0 * omega**2 - 24.0 * inv_dt**2
    beta2 = 12.0 * inv_dt**2 - 12.0 * numerator_damping * omega * inv_dt + omega**2
    return _normalize_section(alpha0, alpha1, alpha2, beta0, beta1, beta2)


def _generalized_correction_section(
    frequency_hz: float,
    numerator_damping: float,
    denominator_damping: float,
    numerator_gamma: float,
    denominator_gamma: float,
    dt: float,
) -> FloatArray:
    omega = 2.0 * np.pi * frequency_hz
    omega_dt = omega * dt
    omega_dt_squared = omega_dt**2
    alpha0 = 1.0 + denominator_damping * omega_dt + denominator_gamma * omega_dt_squared
    alpha1 = -2.0 + omega_dt_squared * (1.0 - 2.0 * denominator_gamma)
    alpha2 = 1.0 - denominator_damping * omega_dt + denominator_gamma * omega_dt_squared
    beta0 = 1.0 + numerator_damping * omega_dt + numerator_gamma * omega_dt_squared
    beta1 = -2.0 + omega_dt_squared * (1.0 - 2.0 * numerator_gamma)
    beta2 = 1.0 - numerator_damping * omega_dt + numerator_gamma * omega_dt_squared
    return _normalize_section(alpha0, alpha1, alpha2, beta0, beta1, beta2)


def _generalized_lowpass_section(
    frequency_hz: float,
    damping: float,
    numerator_gamma: float,
    denominator_gamma: float,
    dt: float,
) -> FloatArray:
    omega = 2.0 * np.pi * frequency_hz
    omega_dt = omega * dt
    omega_dt_squared = omega_dt**2
    alpha0 = 1.0 + damping * omega_dt + denominator_gamma * omega_dt_squared
    alpha1 = -2.0 + omega_dt_squared * (1.0 - 2.0 * denominator_gamma)
    alpha2 = 1.0 - damping * omega_dt + denominator_gamma * omega_dt_squared
    beta0 = numerator_gamma * omega_dt_squared
    beta1 = omega_dt_squared * (1.0 - 2.0 * numerator_gamma)
    beta2 = numerator_gamma * omega_dt_squared
    return _normalize_section(alpha0, alpha1, alpha2, beta0, beta1, beta2)


def _kunugi_2012_sos(sampling_rate_hz: float, parameters: Kunugi2012Parameters) -> FloatArray:
    _validate_parameters(parameters)
    dt = 1.0 / sampling_rate_hz
    sos = np.vstack(
        [
            _paired_base_section(parameters.f0_hz, parameters.f1_hz, dt),
            _paired_half_order_section(parameters.f1_hz, dt),
            _correction_section(
                parameters.correction_hz,
                parameters.correction_numerator_damping,
                parameters.correction_denominator_damping,
                dt,
            ),
            _boxer_thaler_lowpass(
                parameters.lowpass_1_hz,
                parameters.lowpass_1_damping,
                dt,
            ),
            _boxer_thaler_lowpass(
                parameters.lowpass_2_hz,
                parameters.lowpass_2_damping,
                dt,
            ),
            _boxer_thaler_lowpass(
                parameters.lowpass_3_hz,
                parameters.lowpass_3_damping,
                dt,
            ),
        ]
    )
    sos[0, :3] *= parameters.gain
    return sos


def _shared_leading_stages(
    parameters: Kunugi2012Parameters,
    dt: float,
) -> list[FilterStage]:
    """Return A1-A4, the four analog factors the 2012 and low-rate designs share.

    Both designs open with the same period-effect factor and the same three
    7 Hz-region factors, and only diverge from the correction stage onward,
    where the generalized design substitutes its gamma-parameterized
    sections for the fixed Boxer-Thaler ones.
    """
    f0, f1 = parameters.f0_hz, parameters.f1_hz
    return [
        FilterStage("A1: period effect", f0, _first_order_section(0.0, 1.0, f0, dt)),
        FilterStage("A2: 7 Hz factor 1", f1, _first_order_section(1.0, 2.0, f1, dt)),
        FilterStage("A3: 7 Hz factor 2", f1, _first_order_section(4.0, 8.0, f1, dt)),
        FilterStage("A4: 7 Hz factor 3", f1, _first_order_section(0.25, 0.5, f1, dt)),
    ]


def _kunugi_2012_stages(
    sampling_rate_hz: float,
    parameters: Kunugi2012Parameters,
) -> list[FilterStage]:
    """Return the eight named analog factors behind :func:`_kunugi_2012_sos`.

    Commonly labeled A1-A8 in secondary descriptions of Kunugi et al. (2013);
    not independently verified here against the primary journal text. Each
    stage is computed independently with the same per-factor transform used
    to build the combined design (``_paired_base_section`` and
    ``_paired_half_order_section`` are numerically identical to combining
    two independent first-order stages -- see the module's coefficient
    regression tests), so cascading all eight stages plus the gain stage
    reproduces the combined ``sos`` exactly.
    """
    dt = 1.0 / sampling_rate_hz
    fc, lp1, lp2, lp3 = (
        parameters.correction_hz,
        parameters.lowpass_1_hz,
        parameters.lowpass_2_hz,
        parameters.lowpass_3_hz,
    )
    return [
        *_shared_leading_stages(parameters, dt),
        FilterStage(
            "A5: correction",
            fc,
            _correction_section(
                fc,
                parameters.correction_numerator_damping,
                parameters.correction_denominator_damping,
                dt,
            ),
        ),
        FilterStage(
            "A6: low-pass 1", lp1, _boxer_thaler_lowpass(lp1, parameters.lowpass_1_damping, dt)
        ),
        FilterStage(
            "A7: low-pass 2", lp2, _boxer_thaler_lowpass(lp2, parameters.lowpass_2_damping, dt)
        ),
        FilterStage(
            "A8: low-pass 3", lp3, _boxer_thaler_lowpass(lp3, parameters.lowpass_3_damping, dt)
        ),
        _gain_stage(parameters.gain),
    ]


def _kunugi_lowrate_sos(
    sampling_rate_hz: float,
    parameters: Kunugi2012Parameters,
    gammas: LowRateGammaSet,
) -> FloatArray:
    _validate_parameters(parameters)
    _validate_gamma_set(gammas)
    dt = 1.0 / sampling_rate_hz
    sos = np.vstack(
        [
            _paired_base_section(parameters.f0_hz, parameters.f1_hz, dt),
            _paired_half_order_section(parameters.f1_hz, dt),
            _generalized_correction_section(
                parameters.correction_hz,
                parameters.correction_numerator_damping,
                parameters.correction_denominator_damping,
                gammas.correction_numerator,
                gammas.correction_denominator,
                dt,
            ),
            _generalized_lowpass_section(
                parameters.lowpass_1_hz,
                parameters.lowpass_1_damping,
                gammas.lowpass_1_numerator,
                gammas.lowpass_1_denominator,
                dt,
            ),
            _generalized_lowpass_section(
                parameters.lowpass_2_hz,
                parameters.lowpass_2_damping,
                gammas.lowpass_2_numerator,
                gammas.lowpass_2_denominator,
                dt,
            ),
            _generalized_lowpass_section(
                parameters.lowpass_3_hz,
                parameters.lowpass_3_damping,
                gammas.lowpass_3_numerator,
                gammas.lowpass_3_denominator,
                dt,
            ),
        ]
    )
    sos[0, :3] *= parameters.gain
    return sos


def _kunugi_lowrate_stages(
    sampling_rate_hz: float,
    parameters: Kunugi2012Parameters,
    gammas: LowRateGammaSet,
) -> list[FilterStage]:
    """Return the eight named analog factors behind :func:`_kunugi_lowrate_sos`.

    A1-A4 are shared with :func:`_kunugi_2012_stages`; A5-A8 use the
    generalized, gamma-parameterized sections instead of the fixed
    Boxer-Thaler form.
    """
    dt = 1.0 / sampling_rate_hz
    fc, lp1, lp2, lp3 = (
        parameters.correction_hz,
        parameters.lowpass_1_hz,
        parameters.lowpass_2_hz,
        parameters.lowpass_3_hz,
    )
    return [
        *_shared_leading_stages(parameters, dt),
        FilterStage(
            "A5: correction",
            fc,
            _generalized_correction_section(
                fc,
                parameters.correction_numerator_damping,
                parameters.correction_denominator_damping,
                gammas.correction_numerator,
                gammas.correction_denominator,
                dt,
            ),
        ),
        FilterStage(
            "A6: low-pass 1",
            lp1,
            _generalized_lowpass_section(
                lp1,
                parameters.lowpass_1_damping,
                gammas.lowpass_1_numerator,
                gammas.lowpass_1_denominator,
                dt,
            ),
        ),
        FilterStage(
            "A7: low-pass 2",
            lp2,
            _generalized_lowpass_section(
                lp2,
                parameters.lowpass_2_damping,
                gammas.lowpass_2_numerator,
                gammas.lowpass_2_denominator,
                dt,
            ),
        ),
        FilterStage(
            "A8: low-pass 3",
            lp3,
            _generalized_lowpass_section(
                lp3,
                parameters.lowpass_3_damping,
                gammas.lowpass_3_numerator,
                gammas.lowpass_3_denominator,
                dt,
            ),
        ),
        _gain_stage(parameters.gain),
    ]


def _max_pole_radius(sos: npt.NDArray[np.float64]) -> float:
    radii = [np.max(np.abs(np.roots(section[3:]))) for section in sos]
    return float(max(radii, default=0.0))


def _characteristic_frequencies(
    selected: RealtimeFilter,
    parameters: Kunugi2008Parameters | Kunugi2012Parameters,
) -> tuple[float, ...]:
    if selected is RealtimeFilter.KUNUGI_2008:
        assert isinstance(parameters, Kunugi2008Parameters)
        return (parameters.f0_hz, parameters.f1_hz, parameters.f2_hz)
    assert isinstance(parameters, Kunugi2012Parameters)
    return (
        parameters.f0_hz,
        parameters.f1_hz,
        parameters.correction_hz,
        parameters.lowpass_1_hz,
        parameters.lowpass_2_hz,
        parameters.lowpass_3_hz,
    )


def _has_standard_lowrate_frequencies(parameters: Kunugi2012Parameters) -> bool:
    expected = (0.5, 12.0, 20.0, 30.0)
    actual = (
        parameters.correction_hz,
        parameters.lowpass_1_hz,
        parameters.lowpass_2_hz,
        parameters.lowpass_3_hz,
    )
    return all(
        math.isclose(a, e, rel_tol=0.0, abs_tol=1e-12)
        for a, e in zip(actual, expected, strict=True)
    )


def _finish_design(
    selected: RealtimeFilter,
    rate: float,
    parameters: Kunugi2008Parameters | Kunugi2012Parameters,
    sos: FloatArray,
    gamma_parameters: dict[str, float],
    stages: list[FilterStage],
    *,
    check_stability: bool,
    stability_tolerance: float,
) -> RecursiveFilterDesign:
    radius = _max_pole_radius(sos)
    stable = radius < 1.0 - stability_tolerance
    if check_stability and not stable:
        raise UnstableFilterError(
            f"{selected.value} is unstable at {rate:g} Hz "
            f"(maximum pole radius {radius:.12g}). Resample to 100 Hz or select "
            "the generalized low-rate filter with validated gamma values."
        )
    parameter_values = {name: float(value) for name, value in asdict(parameters).items()}
    parameter_values.update(gamma_parameters)
    return RecursiveFilterDesign(
        name=selected.value,
        sampling_rate_hz=rate,
        sos=np.ascontiguousarray(sos),
        parameters=parameter_values,
        characteristic_frequencies_hz=_characteristic_frequencies(selected, parameters),
        max_pole_radius=radius,
        stable=stable,
        stages=tuple(stages),
    )


def design_realtime_filter(
    sampling_rate_hz: float = 100.0,
    *,
    filter_name: str | RealtimeFilter = RealtimeFilter.AUTO,
    parameters: Kunugi2008Parameters | Kunugi2012Parameters | None = None,
    lowrate_gamma_policy: str | LowRateGammaPolicy = LowRateGammaPolicy.PIECEWISE,
    lowrate_gammas: LowRateGammaSet | None = None,
    check_stability: bool = True,
    stability_tolerance: float = 1e-12,
) -> RecursiveFilterDesign:
    """Construct a source-equation recursive filter at a requested sample rate.

    ``auto`` selects the improved 2012 design at 80 Hz and above, where the
    published low-rate gamma table reduces to the original second-integral
    approximation. Below 80 Hz it selects the generalized low-rate design.
    Coefficients are always recomputed from the published equations.

    ``lowrate_gamma_policy`` is used only by the generalized low-rate design.
    Explicit ``lowrate_gammas`` bypass the policy table, while the resulting pole
    locations are still checked unless ``check_stability`` is disabled.
    """
    rate = float(sampling_rate_hz)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("sampling_rate_hz must be finite and greater than zero.")
    if not math.isfinite(stability_tolerance) or stability_tolerance < 0.0:
        raise ValueError("stability_tolerance must be finite and non-negative.")
    requested = RealtimeFilter.parse(filter_name)

    if requested is RealtimeFilter.KUNUGI_2008:
        if parameters is None:
            parameters = Kunugi2008Parameters()
        if not isinstance(parameters, Kunugi2008Parameters):
            raise TypeError("The 2008 filter requires Kunugi2008Parameters.")
        return _finish_design(
            requested,
            rate,
            parameters,
            _kunugi_2008_sos(rate, parameters),
            {},
            _kunugi_2008_stages(rate, parameters),
            check_stability=check_stability,
            stability_tolerance=stability_tolerance,
        )

    if parameters is None:
        parameters = Kunugi2012Parameters()
    if not isinstance(parameters, Kunugi2012Parameters):
        raise TypeError("The 2012 and low-rate filters require Kunugi2012Parameters.")

    selected = requested
    if requested is RealtimeFilter.AUTO:
        selected = (
            RealtimeFilter.KUNUGI_2012
            if rate >= 80.0
            else RealtimeFilter.JP7681907_LOWRATE
        )

    if selected is RealtimeFilter.KUNUGI_2012:
        sos = _kunugi_2012_sos(rate, parameters)
        stages = _kunugi_2012_stages(rate, parameters)
        gamma_parameters: dict[str, float] = {}
    else:
        policy = LowRateGammaPolicy.parse(lowrate_gamma_policy)
        if lowrate_gammas is None:
            if policy.requires_standard_frequencies and not _has_standard_lowrate_frequencies(
                parameters
            ):
                raise ValueError(
                    f"The {policy.value!r} table is published only for correction/low-pass "
                    "frequencies 0.5, 12, 20, and 30 Hz. Supply explicit lowrate_gammas "
                    "or use a constant gamma policy for custom frequencies."
                )
            gammas = published_lowrate_gamma_set(rate, policy=policy)
        else:
            gammas = lowrate_gammas
        sos = _kunugi_lowrate_sos(rate, parameters, gammas)
        stages = _kunugi_lowrate_stages(rate, parameters, gammas)
        gamma_parameters = {
            f"gamma_{name}": float(value) for name, value in asdict(gammas).items()
        }

    return _finish_design(
        selected,
        rate,
        parameters,
        sos,
        gamma_parameters,
        stages,
        check_stability=check_stability,
        stability_tolerance=stability_tolerance,
    )


def _resolve_frequency_grid(
    nyquist_hz: float,
    frequency_hz: npt.ArrayLike | None,
    points: int,
) -> FloatArray:
    if points < 2:
        raise ValueError("points must be at least two.")
    if frequency_hz is None:
        return np.linspace(0.0, nyquist_hz, points, endpoint=False)
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(~np.isfinite(frequency)) or np.any(frequency < 0.0):
        raise ValueError("frequency_hz must be finite and non-negative.")
    if np.any(frequency > nyquist_hz):
        raise ValueError("frequency_hz exceeds the Nyquist frequency.")
    return frequency


def realtime_filter_response(
    design: RecursiveFilterDesign,
    frequency_hz: npt.ArrayLike | None = None,
    *,
    points: int = 4096,
) -> FrequencyResponse:
    """Evaluate a digital real-time filter at physical frequencies."""
    frequency = _resolve_frequency_grid(design.nyquist_hz, frequency_hz, points)
    _, response = sosfreqz(design.sos, worN=frequency, fs=design.sampling_rate_hz)
    return FrequencyResponse(
        frequency_hz=np.asarray(frequency, dtype=np.float64),
        response=np.asarray(response, dtype=np.complex128),
    )


def filter_stage_response(
    design: RecursiveFilterDesign,
    stage: FilterStage,
    frequency_hz: npt.ArrayLike | None = None,
    *,
    points: int = 4096,
) -> FrequencyResponse:
    """Evaluate one named stage of ``design`` on its own, at physical frequencies.

    ``stage`` is normally one of ``design.stages``. The stage's own digital
    response is evaluated in isolation (as a length-one SOS cascade), not the
    combined design.
    """
    frequency = _resolve_frequency_grid(design.nyquist_hz, frequency_hz, points)
    _, response = sosfreqz(
        stage.sos.reshape(1, 6), worN=frequency, fs=design.sampling_rate_hz
    )
    return FrequencyResponse(
        frequency_hz=np.asarray(frequency, dtype=np.float64),
        response=np.asarray(response, dtype=np.complex128),
    )


def kunugi_2012_analog_amplitude(
    frequency_hz: npt.ArrayLike,
    parameters: Kunugi2012Parameters | None = None,
) -> FloatArray:
    """Evaluate the continuous-frequency amplitude of the 2012 approximation.

    This reproduces the analog response comparison in the source paper. It is
    distinct from the sampled digital response, which can be distorted near
    Nyquist.
    """
    p = parameters or Kunugi2012Parameters()
    _validate_parameters(p)
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(~np.isfinite(frequency)) or np.any(frequency < 0.0):
        raise ValueError("frequency_hz must be finite and non-negative.")
    s_inv_magnitude = np.zeros_like(frequency)
    positive = frequency > 0.0
    s_inv_magnitude[positive] = 1.0 / (2.0 * np.pi * frequency[positive])
    q = -1j * s_inv_magnitude

    def first_order(a: float, b: float, f: float) -> npt.NDArray[np.complex128]:
        omega = 2.0 * np.pi * f
        return (a * omega * q + 1.0) / (omega * q + b)

    def damped_ratio(
        f: float,
        numerator_damping: float,
        denominator_damping: float,
    ) -> npt.NDArray[np.complex128]:
        omega = 2.0 * np.pi * f
        q2 = q * q
        return (1.0 + 2.0 * numerator_damping * omega * q + omega**2 * q2) / (
            1.0 + 2.0 * denominator_damping * omega * q + omega**2 * q2
        )

    def lowpass(f: float, damping: float) -> npt.NDArray[np.complex128]:
        omega = 2.0 * np.pi * f
        q2 = q * q
        return omega**2 * q2 / (1.0 + 2.0 * damping * omega * q + omega**2 * q2)

    response = (
        first_order(0.0, 1.0, p.f0_hz)
        * first_order(1.0, 2.0, p.f1_hz)
        * first_order(4.0, 8.0, p.f1_hz)
        * first_order(0.25, 0.5, p.f1_hz)
        * damped_ratio(
            p.correction_hz,
            p.correction_numerator_damping,
            p.correction_denominator_damping,
        )
        * lowpass(p.lowpass_1_hz, p.lowpass_1_damping)
        * lowpass(p.lowpass_2_hz, p.lowpass_2_damping)
        * lowpass(p.lowpass_3_hz, p.lowpass_3_damping)
        * p.gain
    )
    amplitude = np.abs(response)
    amplitude[~positive] = 0.0
    return np.asarray(amplitude, dtype=np.float64)
