"""Shared Parquet checks exercise real serialization and reject the wrong engine root."""
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "parquet_environment_check", Path(__file__).parents[1] / "scripts" / "parquet_environment_check.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_real_nullable_and_empty_roundtrips():
    result = checker.check(Path(checker.pyarrow.__file__).resolve().parent)
    assert result["roundtrip_nullable"] and result["roundtrip_empty"]
    assert result["packages"]["pyarrow"] == checker.pyarrow.__version__
    assert len(result["pyarrow_init_sha256"]) == 64


def test_wrong_engine_root_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside the requested environment"):
        checker.check(tmp_path)
