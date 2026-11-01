"""Shared Parquet checks exercise real serialization and reject the wrong engine root."""
from pathlib import Path

import pytest


from trkfastsim.validation import parquet_environment_check as checker


def test_real_nullable_and_empty_roundtrips():
    pytest.importorskip("jax")    # the check records the JAX version of the environment
    result = checker.check(Path(checker.pyarrow.__file__).resolve().parent)
    assert result["roundtrip_nullable"] and result["roundtrip_empty"]
    assert result["packages"]["pyarrow"] == checker.pyarrow.__version__
    assert len(result["pyarrow_init_sha256"]) == 64


def test_wrong_engine_root_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside the requested environment"):
        checker.check(tmp_path)
