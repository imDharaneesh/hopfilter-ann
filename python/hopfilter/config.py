import dataclasses
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml

from hopfilter.errors import InvalidConfigError

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

ROOT_LOCATION = "config"


def load_config[ConfigT: DataclassInstance](path: Path, config_type: type[ConfigT]) -> ConfigT:
    """Load a YAML file into the frozen dataclass ``config_type``.

    Raises:
        InvalidConfigError: The file is missing, is not a mapping, has unknown or missing
            keys, or holds a value of the wrong type.
    """
    if not path.is_file():
        raise InvalidConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return config_from_mapping({} if raw is None else raw, config_type)


def config_from_mapping[ConfigT: DataclassInstance](
    raw: object, config_type: type[ConfigT], location: str = ROOT_LOCATION
) -> ConfigT:
    """Build ``config_type`` from an already-parsed mapping, validating every key."""
    if not isinstance(raw, Mapping):
        raise InvalidConfigError(f"{location} must be a mapping, got {type(raw).__name__}")
    _check_keys(raw, config_type, location)
    hints = typing.get_type_hints(config_type)
    values = {
        name: _coerce(value, hints[name], f"{location}.{name}") for name, value in raw.items()
    }
    return config_type(**values)


def _check_keys(raw: Mapping[object, object], config_type: type, location: str) -> None:
    fields = dataclasses.fields(config_type)
    known = {field.name for field in fields}
    unknown = sorted(str(key) for key in raw if key not in known)
    if unknown:
        raise InvalidConfigError(f"{location} has unknown keys: {', '.join(unknown)}")
    missing = sorted(
        field.name for field in fields if _is_required(field) and field.name not in raw
    )
    if missing:
        raise InvalidConfigError(f"{location} is missing required keys: {', '.join(missing)}")


def _is_required(field: dataclasses.Field[object]) -> bool:
    return field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING


def _coerce(value: object, expected: object, location: str) -> object:
    if isinstance(expected, typing.TypeAliasType):
        return _coerce(value, expected.__value__, location)
    if isinstance(expected, type) and dataclasses.is_dataclass(expected):
        return config_from_mapping(value, expected, location)
    origin = typing.get_origin(expected)
    if origin is tuple:
        return _coerce_tuple(value, typing.get_args(expected), location)
    if origin is Literal:
        return _coerce_literal(value, typing.get_args(expected), location)
    if expected is float and _is_strict_instance(value, int):
        return float(typing.cast(int, value))
    if isinstance(expected, type) and _is_strict_instance(value, expected):
        return value
    raise InvalidConfigError(
        f"{location} must be {_describe(expected)}, got {type(value).__name__}"
    )


def _coerce_tuple(
    value: object, item_types: tuple[object, ...], location: str
) -> tuple[object, ...]:
    if not isinstance(value, list | tuple):
        raise InvalidConfigError(f"{location} must be a list, got {type(value).__name__}")
    item_type = item_types[0]
    return tuple(
        _coerce(item, item_type, f"{location}[{index}]") for index, item in enumerate(value)
    )


def _coerce_literal(value: object, choices: tuple[object, ...], location: str) -> object:
    if value in choices:
        return value
    allowed = ", ".join(repr(choice) for choice in choices)
    raise InvalidConfigError(f"{location} must be one of {allowed}, got {value!r}")


def _is_strict_instance(value: object, expected: type) -> bool:
    return isinstance(value, expected) and (expected is bool or not isinstance(value, bool))


def _describe(expected: object) -> str:
    return expected.__name__ if isinstance(expected, type) else str(expected)
