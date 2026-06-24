#!/usr/bin/env python3
# Backport vLLM PR #42543 into the installed vllm so Unsloth fast_inference/GRPO
# does not crash at torch.compile graph-split time.
#
# Symptom (vLLM 0.18.0 .. 0.23.0 and main -- the fix is still UNMERGED upstream):
#
#   File ".../vllm/compilation/backends.py", in _decompose_size_nodes
#     graph.graph.erase_node(node)
#   RuntimeError: Tried to erase Node size_1 but it still had 2 users in the
#   graph: {getitem_3: None, getitem_4: None}!
#
# Cause: `_decompose_size_nodes` rewrites the users of an `x.size()` node before
# erasing it, but its rewrite loop only scans top-level `user.args`. When the
# size node is nested inside a Python `slice` (the punica LoRA
# `token_lora_mapping[:x.size(0)]` path that Unsloth's LoRA + vLLM rollout hits),
# those getitem users are never rewritten and `erase_node` raises. This is
# https://github.com/vllm-project/vllm/pull/42543 ("Fix _decompose_size_nodes to
# handle size() in slice bounds"), which the author found "while fine-tuning
# Llama with Unsloth + vLLM".
#
# This patcher APPENDS a guarded block to the installed backends.py that defines
# the PR's recursive arg-rewriter and rebinds the module-global
# `_decompose_size_nodes` the splitter calls. Appending + rebinding (rather than
# in-place surgery) is robust across minor vLLM revisions: it does not depend on
# the exact old function text, only on the names `fx`/`torch` already imported at
# the top of backends.py. Idempotent (skips if the marker is present) and a no-op
# if the upstream fix has already landed (it exposes `_replace_size_in_args`).
#
# Usage:  patch_vllm_compile.py [path/to/vllm/compilation/backends.py]
# Exit 0 = patched or already-good; exit 1 = could not locate/patch (build fails).
import glob
import os
import subprocess
import sys
import sysconfig

_MARKER = "# === UNSLOTH-PATCH: decompose-size-slice (vLLM PR #42543) ==="

_BLOCK = '''

''' + _MARKER + '''
# Backport of https://github.com/vllm-project/vllm/pull/42543 so x.size(dim)
# nodes nested in slice bounds (the punica LoRA token_lora_mapping[:x.size(0)]
# path used by Unsloth GRPO / fast_inference) are rewritten before erase_node,
# instead of crashing with "still had N users". Unmerged upstream as of vLLM
# 0.23.0; remove this block once the fix ships in the pinned vLLM. The rebind at
# the end wins because split_graph() resolves _decompose_size_nodes as a module
# global at call time.
def _replace_size_in_slice(s, node, dims):
    def _sub(bound):
        if isinstance(bound, fx.Node) and bound is node:
            sym_dims = [d for d in dims if isinstance(d, fx.Node)]
            if len(sym_dims) == 1:
                return sym_dims[0]
            return dims[0]
        return bound
    new_start, new_stop, new_step = _sub(s.start), _sub(s.stop), _sub(s.step)
    if new_start is not s.start or new_stop is not s.stop or new_step is not s.step:
        return slice(new_start, new_stop, new_step)
    return s


def _replace_size_in_args(args, node, dims):
    result = []
    for arg in args:
        if isinstance(arg, fx.Node) and arg is node:
            result.extend(dims)
        elif isinstance(arg, slice):
            result.append(_replace_size_in_slice(arg, node, dims))
        elif isinstance(arg, (tuple, list)):
            result.append(type(arg)(_replace_size_in_args(arg, node, dims)))
        else:
            result.append(arg)
    return result


def _unsloth_decompose_size_nodes(graph):
    size_nodes = list(graph.graph.find_nodes(op="call_method", target="size"))
    for node in size_nodes:
        tensor_node = node.args[0]
        ev = tensor_node.meta.get("example_value")
        assert ev is not None, (
            "Tensor node '%s' has no example_value metadata. Cannot decompose "
            "size node '%s'." % (tensor_node.name, node.name)
        )
        dims = []
        with graph.graph.inserting_after(tensor_node):
            for i in range(ev.dim()):
                dim_val = ev.shape[i]
                if isinstance(dim_val, torch.SymInt):
                    dn = graph.graph.call_function(
                        torch.ops.aten.sym_size.int, args=(tensor_node, i)
                    )
                    dn.meta["example_value"] = dim_val
                    dims.append(dn)
                elif isinstance(dim_val, int):
                    dims.append(dim_val)
                else:
                    raise AssertionError(
                        "dim_val is either torch.SymInt or int, got %s for dim "
                        "%d of '%s'" % (type(dim_val), i, node.name)
                    )
        for user in list(node.users):
            if (user.op == "call_function"
                    and user.target is operator.getitem
                    and len(user.args) == 2
                    and user.args[0] is node):
                # getitem(size, idx) -> dims[idx] directly (x.shape[i]); this is
                # the vLLM PR #38360 case, absent in vLLM 0.19.1.
                idx = user.args[1]
                assert isinstance(idx, int), (
                    "Expected literal int index for getitem on size(), got %r"
                    % (idx,)
                )
                user.replace_all_uses_with(dims[idx])
                graph.graph.erase_node(user)
            else:
                # Full size consumed in args, possibly nested in slices/tuples
                # /lists (PR #42543: handles the slice-bound LoRA case).
                user.args = tuple(_replace_size_in_args(user.args, node, dims))
        graph.graph.erase_node(node)


_decompose_size_nodes = _unsloth_decompose_size_nodes
_unsloth_decompose_fix = True
# === /UNSLOTH-PATCH ===
'''


def _find_backends():
    for base in {sysconfig.get_paths().get("purelib"),
                 sysconfig.get_paths().get("platlib")}:
        if not base:
            continue
        p = os.path.join(base, "vllm", "compilation", "backends.py")
        if os.path.isfile(p):
            return p
    hits = glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages",
                                  "vllm", "compilation", "backends.py"))
    return hits[0] if hits else None


def main(argv):
    path = argv[0] if argv else _find_backends()
    if not path or not os.path.isfile(path):
        print("[vllm-patch] backends.py not found; nothing to patch", file=sys.stderr)
        return 1

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if _MARKER in text:
        # Strip our previously appended block (always last) so a re-run installs
        # the current version instead of skipping.
        text = text[:text.find(_MARKER)].rstrip() + "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    if "_replace_size_in_args" in text:
        print(f"[vllm-patch] upstream fix already present: {path}")
        return 0
    if "def _decompose_size_nodes" not in text:
        print(f"[vllm-patch] _decompose_size_nodes not found in {path}; skipping",
              file=sys.stderr)
        return 0  # different vLLM layout -> do not fail the build
    if "fx." not in text:
        print(f"[vllm-patch] 'fx' not referenced in {path}; refusing to patch",
              file=sys.stderr)
        return 1

    with open(path, "a", encoding="utf-8") as f:
        f.write(_BLOCK)

    # Syntax-check the patched module without importing vLLM (no GPU at build).
    rc = subprocess.call([sys.executable, "-m", "py_compile", path])
    if rc != 0:
        print(f"[vllm-patch] py_compile failed after patching {path}", file=sys.stderr)
        return 1
    print(f"[vllm-patch] backported PR #42543 into {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
