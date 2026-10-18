import pytest

from hopfilter.errors import (
    DatasetIntegrityError,
    DatasetNotFoundError,
    EmptyDatasetError,
    HopfilterError,
    IndexNotBuiltError,
    IndexStateError,
    InvalidArgumentError,
    InvalidConfigError,
    SerializationError,
    UnknownIdError,
)


@pytest.mark.parametrize(
    ("error_type", "builtin_base"),
    [
        (InvalidConfigError, ValueError),
        (IndexNotBuiltError, RuntimeError),
        (DatasetNotFoundError, FileNotFoundError),
        (DatasetIntegrityError, RuntimeError),
        (EmptyDatasetError, ValueError),
        (UnknownIdError, KeyError),
        (InvalidArgumentError, ValueError),
        (IndexStateError, RuntimeError),
        (SerializationError, RuntimeError),
    ],
)
def test_error_is_catchable_as_library_and_builtin_type(
    error_type: type[HopfilterError], builtin_base: type[Exception]
) -> None:
    assert issubclass(error_type, HopfilterError)
    assert issubclass(error_type, builtin_base)


def test_error_keeps_message() -> None:
    with pytest.raises(InvalidConfigError, match="M must be positive"):
        raise InvalidConfigError("M must be positive")
