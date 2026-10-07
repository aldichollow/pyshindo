"""Machine-readable status for the algorithms this package implements.

This package's own organizing idea (see the top of the README) is that a
method is one of a few distinct things, and conflating them is a real
mistake: JMA's own published calculation (a *reference* method), a causal
approximation someone else published and this package reconstructs from
its equations (a *published approximation*), a choice this package itself
makes where the primary source leaves one open (a *package extension*), or
a standard numerical technique with no single seismological source to cite
(a *general engineering method*). :func:`method_descriptor` makes that
classification queryable instead of only stated in prose, for tooling,
generated documentation, or a caller who wants to assert "this result only
used reference methods" before trusting it further.

This is a standalone registry, not a field added to
:class:`~pyshindo.models.MeasuredIntensityResult` or any other result
type: adding a new required field to an existing, already-released result
dataclass would be a breaking change for anyone constructing one directly,
and this package treats backward compatibility as something to break
deliberately, not as a side effect of an unrelated feature. Wiring a
``method`` field into every result type is a real possible next step, left
for its own decision rather than bundled in here.

``reference_conditions`` describes what the method's own reference
conditions *are* (for example, "100 Hz sampling"), not whether any one
call met them -- several result types already carry their own
``reference_conditions_met`` field for that per-call question (see
:class:`~pyshindo.models.MeasuredIntensityResult`,
:class:`~pyshindo.long_period.LongPeriodResult`), and duplicating it here
at the method level would only let the two disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MethodStatus(StrEnum):
    """Where one algorithm stands relative to its primary source, if it has one."""

    REFERENCE = "reference"
    """Implements a published authority's own defining calculation directly."""

    PUBLISHED_APPROXIMATION = "published_approximation"
    """Reconstructed from a specific published approximation to a reference method."""

    PACKAGE_EXTENSION = "package_extension"
    """A choice this package makes where the primary source leaves one open."""

    GENERAL_ENGINEERING_METHOD = "general_engineering_method"
    """A standard numerical technique, not itself specific to seismic intensity."""


@dataclass(frozen=True, slots=True)
class MethodDescriptor:
    """What one algorithm is, and what grounds it.

    ``primary_references``, ``deviations``, and ``package_decisions`` are
    free-text, not a closed schema: a citation and a sentence of nuance
    read better in an API response or generated docs than a forced
    structured field would.
    """

    method_id: str
    method_name: str
    status: MethodStatus
    primary_references: tuple[str, ...]
    reference_conditions: str | None
    deviations: tuple[str, ...]
    package_decisions: tuple[str, ...]


_KUNUGI_2008_REFERENCE = (
    'Kunugi, T. et al. (2008), "A Real-Time Processing of Seismic Intensity", '
    "Journal of the Seismological Society of Japan, 60, 243-252. "
    "https://doi.org/10.4294/zisin.60.243"
)
_KUNUGI_2012_REFERENCE = (
    'Kunugi, T. et al. (2013), "An Improved Approximating Filter for Real-Time '
    'Calculation of Seismic Intensity", Journal of the Seismological Society of '
    "Japan, 65, 223-230. https://doi.org/10.4294/zisin.65.223"
)
_KUNUGI_PATENTS_REFERENCE = (
    "Japanese patents JP4229337B2, JP5946067B2, and JP7681907B2 -- see this "
    "repository's PATENTS.md before use or redistribution."
)

_REGISTRY: dict[str, MethodDescriptor] = {
    descriptor.method_id: descriptor
    for descriptor in (
        MethodDescriptor(
            method_id="jma_measured_intensity",
            method_name="JMA instrumental seismic intensity "
            "(frequency-domain reference calculation)",
            status=MethodStatus.REFERENCE,
            primary_references=(
                "Japan Meteorological Agency, 計測震度の算出方法. "
                "https://www.jma.go.jp/jma/kishou/know/jishin/kyoshin/kaisetsu/calc_sindo.html",
            ),
            reference_conditions="100 Hz sampling rate; three mutually orthogonal "
            "acceleration components.",
            deviations=(),
            package_decisions=(
                "The 0.3-second cumulative-duration threshold is selected with an "
                "exact rolling order statistic, not a histogram/binned approximation.",
            ),
        ),
        MethodDescriptor(
            method_id="kunugi_realtime_intensity_2008",
            method_name="Kunugi et al. (2008) causal real-time approximation filter",
            status=MethodStatus.PUBLISHED_APPROXIMATION,
            primary_references=(_KUNUGI_2008_REFERENCE, _KUNUGI_PATENTS_REFERENCE),
            reference_conditions="100 Hz sampling rate.",
            deviations=(
                "Coefficients are re-derived from the published analog prototype "
                "each time, not copied from a fixed table.",
            ),
            package_decisions=(
                "The 60-second rolling threshold uses an exact order statistic, not "
                "a histogram/binned approximation.",
            ),
        ),
        MethodDescriptor(
            method_id="kunugi_realtime_intensity_2012",
            method_name="Kunugi et al. (2013) improved causal real-time approximation filter",
            status=MethodStatus.PUBLISHED_APPROXIMATION,
            primary_references=(_KUNUGI_2012_REFERENCE, _KUNUGI_PATENTS_REFERENCE),
            reference_conditions="100 Hz sampling rate.",
            deviations=(
                "Coefficients are re-derived from the published analog prototype "
                "each time, not copied from a fixed table.",
            ),
            package_decisions=(
                "The 60-second rolling threshold uses an exact order statistic, not "
                "a histogram/binned approximation.",
            ),
        ),
        MethodDescriptor(
            method_id="kunugi_lowrate_realtime_intensity",
            method_name="Generalized low-sampling-rate real-time approximation filter",
            status=MethodStatus.PUBLISHED_APPROXIMATION,
            primary_references=(_KUNUGI_PATENTS_REFERENCE,),
            reference_conditions="Below 100 Hz; reduces to the 2012 filter's own "
            "sections at 100 Hz when gamma = 1/12.",
            deviations=(),
            package_decisions=(
                "The 60-second rolling threshold uses an exact order statistic, not "
                "a histogram/binned approximation.",
            ),
        ),
        MethodDescriptor(
            method_id="jma_long_period_ground_motion_class",
            method_name="JMA long-period ground motion class (長周期地震動階級)",
            status=MethodStatus.REFERENCE,
            primary_references=(
                "Japan Meteorological Agency, 長周期地震動階級および"
                "長周期地震動階級関連解説表について. "
                "https://www.jma.go.jp/jma/kishou/know/jishin/ltpgm_explain/about_level.html",
                "Japan Meteorological Agency, 長周期地震動の観測結果ページの見方. "
                "https://www.data.jma.go.jp/eew/data/ltpgm_explain/about_contents.pdf",
                "大崎順彦 (1994), 新・地震動のスペクトル解析入門 -- the linear-acceleration-method "
                "oscillator solver behind the absolute velocity response. "
                "https://www.data.jma.go.jp/eqev/data/study-panel/tyoshuki_joho_kentokai"
                "/kentokai4/sanko3.pdf",
            ),
            reference_conditions="A 32-oscillator bank over the 1.6-7.8 s period range, "
            "5% damping, horizontal vector composite.",
            deviations=(),
            package_decisions=(
                "Validated against JMA's own published values across 268 stations of "
                "two earthquakes; see docs/validation.md.",
            ),
        ),
        MethodDescriptor(
            method_id="housner_spectrum_intensity",
            method_name="Housner's spectrum intensity (SI value)",
            status=MethodStatus.REFERENCE,
            primary_references=(
                'Housner, G.W. (1959), "Behavior of Structures During Earthquakes", '
                "Journal of the Engineering Mechanics Division, ASCE, 85(EM4), 109-129.",
                'Housner, G.W. (1952), "Spectrum Intensities of Strong Motion Earthquakes".',
                "大崎順彦, 鳥取県道路橋梁設計マニュアル 3-6「スペクトル強度SI値」(式 3-11). "
                "https://www.pref.tottori.lg.jp/secure/198289/dourokkyouryou03-6.pdf",
            ),
            reference_conditions="20% damping; relative velocity response integrated "
            "over the 0.1-2.5 s period range, normalized by 2.4 s.",
            deviations=(
                "No standard integration grid is published; the default 121-point "
                "linear grid and trapezoidal rule were chosen by checking convergence "
                "directly against a 769-point grid.",
            ),
            package_decisions=(),
        ),
        MethodDescriptor(
            method_id="general_elastic_response_spectrum",
            method_name="General elastic single-degree-of-freedom response spectrum (Sd/Sv/PSA)",
            status=MethodStatus.GENERAL_ENGINEERING_METHOD,
            primary_references=(),
            reference_conditions=None,
            deviations=(),
            package_decisions=(
                "Neither damping ratio nor period grid defaults to either the "
                "long-period class's or SI value's own convention -- both are "
                "required arguments rather than an implicit choice between them.",
            ),
        ),
        MethodDescriptor(
            method_id="idw_spatial_interpolation",
            method_name="Shepard-style local inverse-distance-weighted interpolation",
            status=MethodStatus.GENERAL_ENGINEERING_METHOD,
            primary_references=(
                "Donald Shepard, \"A Two-Dimensional Interpolation Function for "
                "Irregularly-Spaced Data,\" Proceedings of the 1968 ACM National "
                "Conference. https://doi.org/10.1145/800186.810616",
            ),
            reference_conditions=None,
            deviations=(),
            package_decisions=(
                "No default search radius: max_distance_km is required unless "
                "allow_extrapolation=True is set explicitly.",
            ),
        ),
        MethodDescriptor(
            method_id="suzuki_2017_pga_idw_preset",
            method_name="Suzuki et al. (2017) PGA interpolation geometry, as an IDW preset",
            status=MethodStatus.PACKAGE_EXTENSION,
            primary_references=(
                "Wataru Suzuki et al., \"Strong motions observed by K-NET and "
                "KiK-net during the 2016 Kumamoto earthquake sequence,\" Earth, "
                "Planets and Space 69, 19 (2017). https://doi.org/10.1186/s40623-017-0604-8",
            ),
            reference_conditions="Four nearest stations within 50 km, "
            "inverse-distance-squared weights.",
            deviations=(
                "minimum_neighbors=4 reads the paper's \"four neighboring stations\" "
                "as a requirement rather than a typical count; the paper does not "
                "state what should happen with fewer nearby stations -- that reading "
                "is this package's own decision, not drawn from the publication.",
            ),
            package_decisions=(),
        ),
    )
}


def method_descriptor(method_id: str) -> MethodDescriptor:
    """Return the descriptor registered for ``method_id``.

    Raises ``ValueError`` naming every registered id, rather than
    ``KeyError``, since a typo'd id is a caller mistake worth a readable
    message instead of a bare lookup failure.
    """
    try:
        return _REGISTRY[method_id]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown method_id {method_id!r}. Known ids: {known}.") from None


def available_method_ids() -> tuple[str, ...]:
    """Return every registered method id, sorted."""
    return tuple(sorted(_REGISTRY))
