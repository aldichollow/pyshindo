"""Warnings and exceptions raised by :mod:`pyshindo`."""

from __future__ import annotations


class PyShindoError(Exception):
    """Base class for package-specific errors."""


class InvalidAccelerationError(PyShindoError, ValueError):
    """Raised when acceleration data have an invalid shape or value."""


class InsufficientDataError(PyShindoError, ValueError):
    """Raised when a record is too short for the requested calculation."""


class UnstableFilterError(PyShindoError, ValueError):
    """Raised when a requested recursive filter is numerically unstable."""


class DataFormatError(PyShindoError, ValueError):
    """Raised when an input record does not match the documented format."""


class PyShindoWarning(UserWarning):
    """Base class for package-specific warnings."""


class NonstandardSamplingRateWarning(PyShindoWarning):
    """Warn that a calculation uses a sampling rate other than 100 Hz."""


class NonstandardProcessingWarning(PyShindoWarning):
    """Warn that optional preprocessing changes the reference calculation."""


class MissingComponentWarning(PyShindoWarning):
    """Warn that fewer than three acceleration components are being used."""


class FractionalDurationWarning(PyShindoWarning):
    """Warn that a duration does not map to an integer number of samples."""
