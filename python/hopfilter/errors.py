class HopfilterError(Exception):
    """Base class for every error raised by hopfilter."""


class InvalidConfigError(HopfilterError, ValueError):
    """A configuration value is missing, malformed, or out of range."""


class InvalidArgumentError(HopfilterError, ValueError):
    """A call violated an API contract, e.g. a vector of the wrong dimension or ``k == 0``."""


class IndexNotBuiltError(HopfilterError, RuntimeError):
    """An index operation was requested before the index was built or loaded."""


class IndexStateError(HopfilterError, RuntimeError):
    """The index cannot perform the operation in its current state, e.g. it is full."""


class SerializationError(HopfilterError, RuntimeError):
    """A saved index is missing, truncated, corrupted, or from an unsupported version."""


class DatasetNotFoundError(HopfilterError, FileNotFoundError):
    """Dataset files are missing locally; run ``hopfilter download`` first."""


class DatasetIntegrityError(HopfilterError, RuntimeError):
    """A downloaded archive failed its checksum or has an unexpected layout."""


class EmptyDatasetError(HopfilterError, ValueError):
    """An operation needs at least one row but received none."""


class UnknownIdError(HopfilterError, KeyError):
    """An id was not seen when the encoder was fitted."""
