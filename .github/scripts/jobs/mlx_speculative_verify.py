# SPDX-License-Identifier: AGPL-3.0-only
"""Validate the MTP runtime candidate on Metal before profiling it."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required = True)
    parser.add_argument("--revision", required = True)
    parser.add_argument(
        "--tiny", action = "store_true", help = "run the explicit CPU-simulation portability checks"
    )
    parser.add_argument("--native-head-control", action = "store_true")
    args = parser.parse_args()
    output = Path(args.out).resolve()
    result = {"finished": False, "passed": False, "revision": args.revision}
    scratch = Path(os.environ.get("RUNNER_TEMP", "temp"))
    scratch.mkdir(parents = True, exist_ok = True)
    try:
        os.environ["UNSLOTH_IS_PRESENT"] = "1"
        os.environ.pop("UNSLOTH_FORCE_GPU_PATH", None)
        installed = json.loads(
            importlib.metadata.distribution("unsloth_zoo").read_text("direct_url.json")
        )
        assert installed.get("vcs_info", {}).get("commit_id") == args.revision, installed
        result["installed_revision"] = installed["vcs_info"]["commit_id"]
        versions = {}
        for name in ("torch", "mlx", "mlx-lm", "mlx-vlm", "unsloth_zoo"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        result.update(mode = "cpu_simulation" if args.tiny else "metal", versions = versions)
        if args.tiny:
            result["device"] = "cpu"
        else:
            import mlx.core as mx

            assert mx.metal.is_available(), "Metal unavailable"
            mx.set_default_device(mx.gpu)
            result["device"] = str(mx.default_device())
        with tempfile.TemporaryDirectory(prefix = "mtp-verify-", dir = scratch) as folder:
            root = Path(folder)
            repo = root / "zoo"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--no-checkout",
                    "--depth=1",
                    "--branch",
                    "fix-mlx-draft-cost",
                    "https://github.com/danielhanchen/unsloth-zoo-staging.git",
                    str(repo),
                ],
                check = True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "fetch", "--depth=1", "origin", args.revision], check = True
            )
            subprocess.run(
                ["git", "-C", str(repo), "checkout", "--detach", args.revision], check = True
            )
            head = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text = True
            ).strip()
            assert head == args.revision, (head, args.revision)
            report = root / "tests.xml"
            bootstrap = """import sys
from pathlib import Path
mode = sys.argv.pop(1)
if mode == 'cpu':
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()
    import torch
    assert not torch.cuda.is_available()
else:
    import mlx.core as mx
    assert mx.metal.is_available()
    mx.set_default_device(mx.gpu)
import unsloth_zoo.mlx.speculative as source
assert Path(source.__file__).resolve() == Path.cwd() / 'unsloth_zoo/mlx/speculative.py'
print('VERIFIED_SOURCE', source.__file__, mode, flush=True)
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
"""
            command = [
                sys.executable,
                "-c",
                bootstrap,
                "cpu" if args.tiny else "metal",
                "-q",
                "--tb=short",
                f"--junitxml={report}",
                "tests/test_mlx_speculative.py",
            ]
            if args.tiny:
                command.extend(["-k", "not quantized_decode_keeps_each_rows_padding_mask"])
            else:
                command.extend(
                    [
                        "tests/test_mlx_speculative_metal.py::test_the_shortlist_head_takes_the_tables_own_best_of_its_coarse_copys_few",
                        "tests/test_mlx_speculative_metal.py::test_mtp_rounds_match_solo_decoding",
                        "tests/test_mlx_speculative_metal.py::test_mtp_catches_up_after_plain_windows_and_copies",
                        "tests/test_mlx_speculative_metal.py::test_a_processor_that_settles_speculates_and_is_told_each_committed_history",
                        "tests/test_mlx_speculative_metal.py::test_an_mtp_heads_cache_holds_only_target_hidden_pairs_however_its_drafts_fared",
                        "tests/test_mlx_speculative_metal.py::test_a_lazy_drafter_weighs_its_checkpoint_and_holds_a_started_row_without_allocating[None]",
                    ]
                )
            env = dict(
                os.environ,
                PYTHONPATH = os.pathsep.join([str(repo), str(repo / "tests")]),
                HF_DOWNLOAD_OK = "1",
                UNSLOTH_DISABLE_AUTO_UPDATES = "1",
            )
            if args.native_head_control:
                assert not args.tiny, "native-head negative control requires Metal"
                source = repo / "unsloth_zoo/mlx/speculative.py"
                original = source.read_text(encoding = "utf-8")
                condition = "if width <= _SHORTLIST_COLUMNS or width % _SHORTLIST_GROUP:"
                assert original.count(condition) == 1
                control_report = root / "negative.xml"
                source.write_text(
                    original.replace(condition, "if width % _SHORTLIST_GROUP:"), encoding = "utf-8"
                )
                try:
                    control = subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            bootstrap,
                            "metal",
                            "-q",
                            "--tb=short",
                            f"--junitxml={control_report}",
                            "tests/test_mlx_speculative.py::test_full_width_draft_projection_uses_the_native_table",
                        ],
                        cwd = repo,
                        env = env,
                        check = False,
                    )
                finally:
                    source.write_text(original, encoding = "utf-8")
                controls = ET.parse(control_report).findall(".//testcase")
                failed = [case.get("name") for case in controls if case.find("failure") is not None]
                assert control.returncode == 1 and len(controls) == 4
                assert len(failed) == 3 and not any(
                    case.find("error") is not None or case.find("skipped") is not None
                    for case in controls
                )
                assert {name.rsplit("[", 1)[1].rstrip("]") for name in failed} == {
                    "64",
                    "1024",
                    "2048",
                }
                result["native_head_negative_control"] = {
                    "failed": failed,
                    "passed": 1,
                    "skipped": 0,
                }
            proc = subprocess.run(command, cwd = repo, env = env, check = False)
            cases = ET.parse(report).findall(".//testcase")
            result["tests"] = [
                {
                    "id": f"{case.get('classname')}::{case.get('name')}",
                    "outcome": "failed"
                    if case.find("failure") is not None or case.find("error") is not None
                    else "skipped"
                    if case.find("skipped") is not None
                    else "passed",
                }
                for case in cases
            ]
            required = (
                "test_folded_replay_is_not",
                "test_closing_a_plain_stream",
                "test_mtp_defers_prompt",
            )
            gates = all(
                any(name in case["id"] and case["outcome"] == "passed" for case in result["tests"])
                for name in required
            )
            result["passed"] = (
                proc.returncode == 0
                and bool(cases)
                and gates
                and not any(case["outcome"] == "failed" for case in result["tests"])
            )
            if not args.tiny:
                result["passed"] = result["passed"] and all(
                    case["outcome"] == "passed" for case in result["tests"]
                )
            result["pytest_exit"] = proc.returncode
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        result["finished"] = True
        output.write_text(json.dumps(result, indent = 2), encoding = "utf-8")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
