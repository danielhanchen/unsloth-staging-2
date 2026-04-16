"""Tests for _name_is_local_path: unusual paths, edge cases."""
import os
import tempfile
from test_helpers import get_fn

fn = get_fn("_name_is_local_path")


def test_none():
    assert fn(None) is False


def test_empty_string():
    assert fn("") is False


def test_hub_model_id():
    assert fn("meta-llama/Llama-3-8B") is False


def test_nonexistent_path():
    assert fn("/nonexistent/path/to/model") is False


def test_real_directory():
    with tempfile.TemporaryDirectory() as d:
        assert fn(d) is True


def test_file_not_dir():
    with tempfile.NamedTemporaryFile() as f:
        assert fn(f.name) is False


def test_integer_input():
    assert fn(42) is False


def test_path_with_spaces():
    with tempfile.TemporaryDirectory() as parent:
        spaced = os.path.join(parent, "model dir")
        os.makedirs(spaced)
        assert fn(spaced) is True


def test_decorated_path_fails():
    with tempfile.TemporaryDirectory() as d:
        assert fn(f"{d} (variant='default')") is False


def test_dot_path():
    assert fn(".") is True
