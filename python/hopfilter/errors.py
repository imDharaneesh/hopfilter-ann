class HopfilterError(Exception):
    """Base class for every error raised by hopfilter."""


class InvalidConfigError(HopfilterError, ValueError):
    """A configuration value is missing, malformed, or out of range."""


class IndexNotBuiltError(HopfilterError, RuntimeError):
    """An index operation was requested before the index was built or loaded."""


class DatasetNotFoundError(HopfilterError, FileNotFoundError):
    """Dataset files are missing locally; run ``hopfilter download`` first."""


class DatasetIntegrityError(HopfilterError, RuntimeError):
    """A downloaded archive failed its checksum or has an unexpected layout."""
