"""Tests for scan_dataset entry points, show_row rendering, and format-alias
handling added in the review-round fixes to dataset_none_detect."""

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


# ---------------------------------------------------------------------------
# Minimal dataset mock (pyarrow cannot store non-list messages alongside list messages)
# ---------------------------------------------------------------------------


class _MockDS:
    def __init__(self, rows, columns):
        self._rows = rows
        self.column_names = columns

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]


# ---------------------------------------------------------------------------
# show_row for non-list conversation columns (Fix L)
# ---------------------------------------------------------------------------


def test_show_row_non_list_messages_none_is_displayed():
    ds = _MockDS([{"messages": None, "label": "a"}], ["messages", "label"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.show_row(ds, [0], fmt="chatml")
    out = buf.getvalue()
    assert "messages:" in out
    assert "[None]" in out
    assert "<< BAD" in out


def test_show_row_non_list_messages_string_is_displayed():
    ds = _MockDS([{"messages": "oops not a list", "label": "b"}],
                 ["messages", "label"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.show_row(ds, [0], fmt="chatml")
    out = buf.getvalue()
    assert "messages:" in out
    assert "[invalid_type]" in out
    assert "<< BAD" in out


def test_show_row_non_list_messages_dict_is_displayed():
    ds = _MockDS([{"messages": {"stray": "dict"}, "label": "c"}],
                 ["messages", "label"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.show_row(ds, [0], fmt="chatml")
    out = buf.getvalue()
    assert "messages:" in out
    assert "[invalid_type]" in out


# ---------------------------------------------------------------------------
# scan_dataset on zero-row datasets with explicit format (Fix M)
# ---------------------------------------------------------------------------


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


def test_empty_sharegpt_dataset_without_conv_column_raises():
    ds = Dataset.from_list([{"instruction": "x", "output": "y"}]).select([])
    with pytest.raises(ValueError):
        mod.scan_dataset(ds, fmt="sharegpt")


def test_empty_chatml_dataset_without_conv_column_raises():
    ds = Dataset.from_list([{"instruction": "x", "output": "y"}]).select([])
    with pytest.raises(ValueError):
        mod.scan_dataset(ds, fmt="chatml")


def test_empty_auto_mode_returns_unknown_without_raising():
    ds = Dataset.from_list([{"x": 1}]).select([])
    stats = mod.scan_dataset(ds)
    assert stats["format"] == "unknown"
    assert stats["total_rows"] == 0


# ---------------------------------------------------------------------------
# gpt-oss alias handling (Fix O)
# ---------------------------------------------------------------------------


def test_gpt_oss_alias_normalized_in_scan_dataset():
    ds = Dataset.from_list([{"messages": [
        {"role": "developer", "content": "sys"},
        {"role": "user", "content": None},
    ]}])
    stats = mod.scan_dataset(ds, fmt="gpt-oss")
    assert stats["format"] == "gptoss"
    assert stats["bad_row_indices"] == [0]


def test_gpt_oss_alias_cli_choice_is_accepted_by_argparse():
    # Ensure the CLI argparse choices include the alias (regression guard).
    assert "gpt-oss" in mod.FORMAT_ALIASES
    assert mod.FORMAT_ALIASES["gpt-oss"] == "gptoss"


def test_unknown_alias_still_raises():
    ds = Dataset.from_list([{"messages": [{"role": "user", "content": "hi"}]}])
    with pytest.raises(ValueError):
        mod.scan_dataset(ds, fmt="not-a-real-alias")


# ---------------------------------------------------------------------------
# find_none_chatml error wording (Fix P)
# ---------------------------------------------------------------------------


def test_find_none_chatml_names_explicit_missing_column():
    ds = Dataset.from_list([{"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]}])
    with pytest.raises(ValueError) as exc:
        mod.find_none_chatml(ds, col="nope_custom")
    msg = str(exc.value)
    assert "nope_custom" in msg
    assert "Available columns" in msg or "column" in msg.lower()


def test_find_none_chatml_auto_probe_failure_lists_defaults():
    ds = Dataset.from_list([{"foo": "bar"}])
    with pytest.raises(ValueError) as exc:
        mod.find_none_chatml(ds)
    assert "messages" in str(exc.value) or "conversations" in str(exc.value)


# ---------------------------------------------------------------------------
# _truncate_repr boundary behavior (Fix N)
# ---------------------------------------------------------------------------


def test_truncate_repr_short_string_unchanged():
    assert mod._truncate_repr("hello") == repr("hello")


def test_truncate_repr_long_string_truncates_before_repr():
    # A 100k-character string: _truncate_repr must truncate BEFORE repr() so
    # it never allocates the full repr of the 100k-char input.
    huge = "x" * 100_000
    out = mod._truncate_repr(huge, max_len=500)
    assert len(out) < 600
    assert out.endswith("...")


def test_truncate_repr_non_string_object_also_capped():
    big_list = list(range(10_000))
    out = mod._truncate_repr(big_list, max_len=500)
    assert len(out) < 600


# ---------------------------------------------------------------------------
# SPDX header present (Fix N)
# ---------------------------------------------------------------------------


def test_module_has_spdx_license_header():
    path = REPO_ROOT / "studio/backend/utils/datasets/dataset_none_detect.py"
    header = path.read_text(encoding="utf-8").splitlines()[:3]
    assert any("SPDX-License-Identifier" in line for line in header)
    assert any("Copyright" in line for line in header)
