"""Behaviour tests for studio/backend/utils/datasets/dataset_none_detect."""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "studio" / "backend" / "utils" / "datasets"))

import dataset_none_detect as mod  # noqa: E402


class _MockDS:
    """Datasets.Dataset stand-in; pyarrow rejects mixed list/non-list columns."""

    def __init__(self, rows, columns):
        self._rows = rows
        self.column_names = columns

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]


# show_row on rows where the conversation column is not a list ---------------


@pytest.mark.parametrize(
    "bad_value, expected_label",
    [
        (None, "[None]"),
        ("oops not a list", "[invalid_type]"),
        ({"stray": "dict"}, "[invalid_type]"),
    ],
)
def test_show_row_non_list_conversation_is_displayed(bad_value, expected_label):
    ds = _MockDS([{"messages": bad_value, "label": "x"}], ["messages", "label"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.show_row(ds, [0], fmt="chatml")
    out = buf.getvalue()
    assert "messages:" in out
    assert expected_label in out
    assert "<< BAD" in out


# Zero-row datasets ---------------------------------------------------------


def test_empty_alpaca_dataset_missing_columns_raises():
    ds = Dataset.from_list([{"prompt": "x"}]).select([])
    with pytest.raises(ValueError) as exc:
        mod.scan_dataset(ds, fmt="alpaca")
    assert "instruction" in str(exc.value) or "output" in str(exc.value)


def test_empty_alpaca_dataset_with_columns_is_clean():
    ds = Dataset.from_list([{"instruction": "x", "output": "y"}]).select([])
    stats = mod.scan_dataset(ds, fmt="alpaca")
    assert stats["format"] == "alpaca"
    assert stats["total_rows"] == 0
    assert stats["bad_row_indices"] == []


@pytest.mark.parametrize("fmt", ["chatml", "sharegpt"])
def test_empty_conversation_format_without_column_raises(fmt):
    ds = Dataset.from_list([{"instruction": "x", "output": "y"}]).select([])
    with pytest.raises(ValueError):
        mod.scan_dataset(ds, fmt=fmt)


def test_empty_auto_mode_returns_unknown_without_raising():
    ds = Dataset.from_list([{"x": 1}]).select([])
    stats = mod.scan_dataset(ds)
    assert stats["format"] == "unknown"
    assert stats["total_rows"] == 0


# gpt-oss alias -------------------------------------------------------------


def test_gpt_oss_alias_normalized_in_scan_dataset():
    ds = Dataset.from_list([{"messages": [
        {"role": "developer", "content": "sys"},
        {"role": "user", "content": None},
    ]}])
    stats = mod.scan_dataset(ds, fmt="gpt-oss")
    assert stats["format"] == "gptoss"
    assert stats["bad_row_indices"] == [0]


def test_gpt_oss_alias_is_registered():
    assert mod.FORMAT_ALIASES.get("gpt-oss") == "gptoss"


def test_unknown_alias_raises():
    ds = Dataset.from_list([{"messages": [{"role": "user", "content": "hi"}]}])
    with pytest.raises(ValueError):
        mod.scan_dataset(ds, fmt="not-a-real-alias")


# find_none_chatml error wording --------------------------------------------


def test_find_none_chatml_names_explicit_missing_column():
    ds = Dataset.from_list([{"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]}])
    with pytest.raises(ValueError) as exc:
        mod.find_none_chatml(ds, col="nope_custom")
    msg = str(exc.value)
    assert "nope_custom" in msg
    assert "Available columns" in msg or "not found" in msg.lower()


def test_find_none_chatml_auto_probe_failure_lists_defaults():
    ds = Dataset.from_list([{"foo": "bar"}])
    with pytest.raises(ValueError) as exc:
        mod.find_none_chatml(ds)
    assert "messages" in str(exc.value) or "conversations" in str(exc.value)


# _truncate_repr ------------------------------------------------------------


def test_truncate_repr_short_string_unchanged():
    assert mod._truncate_repr("hello") == repr("hello")


def test_truncate_repr_long_string_truncates_before_repr():
    out = mod._truncate_repr("x" * 100_000, max_len=500)
    assert len(out) < 600
    assert out.endswith("...")


def test_truncate_repr_non_string_object_also_capped():
    out = mod._truncate_repr(list(range(10_000)), max_len=500)
    assert len(out) < 600


# Module-level invariants ---------------------------------------------------


def test_module_has_spdx_license_header():
    path = REPO_ROOT / "studio/backend/utils/datasets/dataset_none_detect.py"
    header = path.read_text(encoding="utf-8").splitlines()[:3]
    assert any("SPDX-License-Identifier" in line for line in header)
    assert any("Copyright" in line for line in header)
