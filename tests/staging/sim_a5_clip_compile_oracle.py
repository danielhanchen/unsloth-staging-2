#!/usr/bin/env python3
"""A5: clip-mode + mx.compile-eligibility oracle for unslothai/unsloth#7917.

The decision-making simulation. It answers three things against the REAL installed
unsloth_zoo source (not a restatement of it):

  1. What clip mode does each Studio configuration resolve to?
  2. Does global-norm clipping make MLX VLM runs ineligible for mx.compile?
  3. Does report_grad_norm=True with max_grad_norm=0 compute a gradient norm,
     i.e. is there a way to populate the chart without changing clipping?

unsloth_zoo.mlx.trainer cannot be imported on Linux (it does `import mlx.core` at
module scope, and the workspace venv's unsloth_zoo/__init__ is separately broken by
a torch/torchao mismatch). So the pure functions are extracted from the source with
ast and exec'd in a clean namespace. Everything asserted here is re-confirmed at
runtime on a real Mac in Tier D.
"""

import ast
import sys
from pathlib import Path

def _locate_zoo_trainer():
    """Find unsloth_zoo/mlx/trainer.py on disk without importing the package."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    import importlib.util
    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("FATAL: unsloth_zoo not installed")
    for root in spec.submodule_search_locations:
        cand = Path(root) / "mlx" / "trainer.py"
        if cand.exists():
            return cand
    raise SystemExit("FATAL: unsloth_zoo/mlx/trainer.py not found")


ZOO = _locate_zoo_trainer()
print(f"zoo trainer source: {ZOO}")

SRC = ZOO.read_text(encoding="utf-8")
TREE = ast.parse(SRC)
FAILURES = []


def check(label, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got {got!r}, want {want!r}")
    if not ok:
        FAILURES.append(label)


def extract_function(name):
    """exec a single top-level function from the zoo source in a clean namespace."""
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(ZOO), "exec"), ns)
            return ns[name]
    raise SystemExit(f"FATAL: {name} not found in {ZOO}")


class Args:
    """Stand-in for MLXTrainingConfig; the resolver only getattrs these."""

    def __init__(self, norm=0.0, value=None, leaf=None, grad_accum=4, report=False):
        self.max_grad_norm = norm
        self.max_grad_value = value
        self.max_grad_leaf_norm = leaf
        self.gradient_accumulation_steps = grad_accum
        self.report_grad_norm = report


# --------------------------------------------------------------------------
print("\n== 1. Clip-mode resolution (unsloth_zoo._resolve_mlx_grad_clipping) ==")
resolve = extract_function("_resolve_mlx_grad_clipping")

CASES = [
    ("BEFORE PR: worker hardcoded 0.0, siblings unset", Args(norm=0.0), "leaf_norm"),
    ("AFTER PR:  unset resolves to 1.0", Args(norm=1.0), "global_norm"),
    ("explicit 0 from an old cached frontend", Args(norm=0.0), "leaf_norm"),
    ("positive leaf outranks global", Args(norm=1.0, leaf=1.3), "leaf_norm"),
    ("positive value outranks global", Args(norm=1.0, value=3.0), "value"),
    ("value outranks both", Args(norm=1.0, value=3.0, leaf=1.3), "value"),
    ("leaf=0 does NOT suppress global", Args(norm=1.0, leaf=0.0), "global_norm"),
    ("explicit zeros disable clipping entirely", Args(norm=0.0, leaf=0.0), "none"),
    ("None norm behaves as 0.0", Args(norm=None), "leaf_norm"),
]
for label, args, want_mode in CASES:
    check(label, resolve(args)[-1], want_mode)


# --------------------------------------------------------------------------
print("\n== 2. mx.compile eligibility (VLM planner) ==")
# The gate lives in _plan_single_process_vlm_shapes. Pull its exact condition out of
# the AST rather than hand-copying it, so this sim cannot silently drift from the zoo.
gate = None
for node in ast.walk(TREE):
    if isinstance(node, ast.FunctionDef) and node.name == "_plan_single_process_vlm_shapes":
        for sub in ast.walk(node):
            if not isinstance(sub, ast.If):
                continue
            for ret in ast.walk(sub):
                if isinstance(ret, ast.Constant) and ret.value == "compile_ineligible_global_norm":
                    gate = sub
                    break
            if gate is not None:
                break
if gate is None:
    raise SystemExit("FATAL: compile_ineligible_global_norm gate not found; zoo changed shape")

gate_src = ast.unparse(gate.test)
print(f"  gate condition, verbatim from the zoo AST: {gate_src}")
check("gate mentions max_grad_norm", "max_grad_norm" in gate_src, True)
check("gate mentions gradient_accumulation_steps", "gradient_accumulation_steps" in gate_src, True)


def compile_ineligible(args, world=1):
    """Evaluate the zoo's own gate expression with the resolved clip threshold."""
    resolved_norm = resolve(args)[0]
    return eval(  # noqa: S307 - expression comes from the zoo source itself
        gate_src,
        {},
        {
            "distributed_world_size": world,
            "max_grad_norm": resolved_norm,
            "args": args,
        },
    )


print("  -- VLM, single device, Studio's default gradient_accumulation_steps=4 --")
check("BEFORE PR (leaf_norm)  -> compile INELIGIBLE?", compile_ineligible(Args(norm=0.0)), False)
check("AFTER PR  (global_norm) -> compile INELIGIBLE?", compile_ineligible(Args(norm=1.0)), True)
check("ALTERNATIVE (report_grad_norm, norm=0) -> INELIGIBLE?",
      compile_ineligible(Args(norm=0.0, report=True)), False)
print("  -- boundary conditions --")
check("global_norm but grad_accum=1 -> INELIGIBLE?", compile_ineligible(Args(norm=1.0, grad_accum=1)), False)
check("global_norm but distributed (world=2) -> INELIGIBLE?", compile_ineligible(Args(norm=1.0), world=2), False)
check("explicit leaf clip (mode leaf_norm) -> INELIGIBLE?", compile_ineligible(Args(norm=1.0, leaf=1.3)), False)

# Scope check: the TEXT planner must NOT carry the same gate, or the regression would
# hit every MLX run rather than only VLM ones.
text_fn = next(n for n in ast.walk(TREE)
               if isinstance(n, ast.FunctionDef) and n.name == "_plan_single_process_text_shapes")
text_has_gate = any(
    isinstance(c, ast.Constant) and c.value == "compile_ineligible_global_norm"
    for c in ast.walk(text_fn)
)
check("text planner also gates on global norm (would widen the blast radius)", text_has_gate, False)


# --------------------------------------------------------------------------
print("\n== 3. Is there a chart fix that does NOT change clipping? ==")
# _compute_report_norm decides whether the trainer computes a norm when it is not
# already clipping globally. Read the assignment straight out of the source.
report_expr = None
for node in ast.walk(TREE):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        t = node.targets[0]
        if isinstance(t, ast.Name) and t.id == "_compute_report_norm":
            report_expr = ast.unparse(node.value)
            break
if report_expr is None:
    raise SystemExit("FATAL: _compute_report_norm not found; zoo changed shape")

print(f"  _compute_report_norm = {report_expr}")


def computes_report_norm(args):
    resolved_norm = resolve(args)[0]
    return bool(eval(  # noqa: S307 - expression comes from the zoo source itself
        report_expr,
        {},
        {"_report_grad_norm": bool(args.report_grad_norm), "max_grad_norm": resolved_norm},
    ))


check("BEFORE PR (no report flag) -> norm computed?", computes_report_norm(Args(norm=0.0)), False)
check("ALTERNATIVE report_grad_norm=True, norm=0 -> norm computed?",
      computes_report_norm(Args(norm=0.0, report=True)), True)
check("report_grad_norm=True but norm=1.0 -> redundant (clip already reports)",
      computes_report_norm(Args(norm=1.0, report=True)), False)

# report_grad_norm must actually exist on the config this zoo ships, or the
# alternative is not implementable without a zoo bump.
has_field = f"\n    report_grad_norm:" in SRC
check("MLXTrainingConfig exposes report_grad_norm", has_field, True)


# --------------------------------------------------------------------------
print("\n" + "=" * 70)
if FAILURES:
    print(f"A5 FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
    raise SystemExit(1)
print("A5 PASSED. Conclusions:")
print("  - The PR flips MLX from leaf_norm to global_norm clipping.")
print("  - That makes single-device MLX *VLM* runs with grad_accum>1 skip mx.compile.")
print("  - Text runs are unaffected by the compile gate.")
print("  - report_grad_norm=True with max_grad_norm=0 computes the same norm while")
print("    leaving clipping semantics AND compile eligibility untouched.")
raise SystemExit(0)
