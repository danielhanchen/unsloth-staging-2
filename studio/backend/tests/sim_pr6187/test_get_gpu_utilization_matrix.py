"""Cartesian [OS] x [GPU] safety for get_gpu_utilization().

detect_hardware() routes to DeviceType.MLX only when
`is_apple_silicon() and _has_mlx()`. So across the full
[Windows, Linux, WSL, Mac] x [NVIDIA, AMD, CPU] matrix, exactly one cell
(Apple Silicon + MLX) reaches the code PR #6187 changed. These tests:

  A. drive the MLX branch and assert the response dict shape is invariant and
     temp/power flow through (value present, and None when sensors are absent
     i.e. identical to the pre-PR behavior).
  B. structurally prove `apple` is referenced ONLY inside the MLX branch, so
     every other device path is byte-identical to before the PR.
  C. assert the MLX gate (is_apple_silicon) is true only on Darwin+arm64.
"""

import ast
import inspect
import textwrap

import pytest

CANONICAL_KEYS = {
    "available", "backend", "gpu_utilization_pct", "temperature_c",
    "vram_used_gb", "vram_total_gb", "vram_utilization_pct",
    "power_draw_w", "power_limit_w", "power_utilization_pct",
}


@pytest.fixture
def hw():
    from utils.hardware import hardware as mod
    return mod


# ---------- A. MLX branch: shape invariant + sensor flow-through ----------

def _drive_mlx(hw, apple, monkeypatch, temp, power):
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.MLX)
    monkeypatch.setattr(
        hw, "_read_apple_gpu_stats",
        lambda: {"utilization_pct": 37.0, "vram_used_bytes": 8 * 1024**3},
    )
    monkeypatch.setattr(apple, "read_gpu_temperature_c", lambda: temp)
    monkeypatch.setattr(apple, "read_gpu_power_w", lambda: power)
    return hw.get_gpu_utilization()


def test_mlx_with_sensors_present(hw, apple, monkeypatch):
    res = _drive_mlx(hw, apple, monkeypatch, temp=85.0, power=26.5)
    assert set(res.keys()) == CANONICAL_KEYS
    assert res["available"] is True
    assert res["temperature_c"] == 85.0
    assert res["power_draw_w"] == 26.5
    assert res["gpu_utilization_pct"] == 37.0


def test_mlx_with_sensors_absent_matches_pre_pr(hw, apple, monkeypatch):
    # Sensors unavailable (locked-down host / VM): temp & power are None,
    # exactly the pre-PR behavior. The monitor stays available.
    res = _drive_mlx(hw, apple, monkeypatch, temp=None, power=None)
    assert set(res.keys()) == CANONICAL_KEYS
    assert res["available"] is True
    assert res["temperature_c"] is None
    assert res["power_draw_w"] is None
    assert res["vram_total_gb"] > 0


def test_mlx_dict_shape_constant_regardless_of_sensor_state(hw, apple, monkeypatch):
    shapes = []
    for t, p in [(None, None), (90.1, None), (None, 12.0), (70.0, 30.0)]:
        shapes.append(tuple(sorted(_drive_mlx(hw, apple, monkeypatch, t, p))))
    assert len(set(shapes)) == 1  # identical key set in every case


# ---------- B. structural proof: apple referenced only in the MLX branch ----------

def test_apple_only_referenced_inside_mlx_branch(hw):
    src = textwrap.dedent(inspect.getsource(hw.get_gpu_utilization))
    tree = ast.parse(src)
    func = tree.body[0]

    # Find the `if device == DeviceType.MLX:` block and its line span.
    mlx_if = None
    for node in ast.walk(func):
        if isinstance(node, ast.If):
            t = node.test
            if (isinstance(t, ast.Compare)
                    and isinstance(t.left, ast.Name) and t.left.id == "device"
                    and isinstance(t.comparators[0], ast.Attribute)
                    and t.comparators[0].attr == "MLX"):
                mlx_if = node
                break
    assert mlx_if is not None, "could not locate the MLX branch"
    lo, hi = mlx_if.lineno, mlx_if.end_lineno

    # Every reference to the name `apple` must fall inside that span.
    apple_lines = [
        n.lineno for n in ast.walk(func)
        if (isinstance(n, ast.Name) and n.id == "apple")
        or (isinstance(n, ast.alias) and n.name == "apple")
    ]
    assert apple_lines, "expected apple to be referenced in the MLX branch"
    outside = [ln for ln in apple_lines if not (lo <= ln <= hi)]
    assert outside == [], f"apple referenced outside MLX branch at lines {outside}"


# ---------- C. MLX gate is Darwin+arm64 only ----------

@pytest.mark.parametrize("system,machine,expected", [
    ("Darwin", "arm64", True),
    ("Darwin", "x86_64", False),   # Intel Mac -> never MLX
    ("Linux", "x86_64", False),    # Linux NVIDIA/AMD/CPU
    ("Linux", "aarch64", False),   # ARM Linux, still not Darwin
    ("Windows", "AMD64", False),   # Windows NVIDIA/AMD/CPU
    ("Linux", "arm64", False),     # WSL/other ARM, not Darwin
])
def test_is_apple_silicon_gate(hw, monkeypatch, system, machine, expected):
    monkeypatch.setattr(hw.platform, "system", lambda: system)
    monkeypatch.setattr(hw.platform, "machine", lambda: machine)
    assert hw.is_apple_silicon() is expected
