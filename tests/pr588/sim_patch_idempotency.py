"""Sim test: monkey-patches are idempotent across repeated calls.

Note: patch_gemma4_vllm_lora_support transitively imports unsloth_zoo.vllm_lora_worker_manager
which hard-imports many vllm.lora.* internals. We skip the full call-tests if vllm isn't
installed, and additionally test idempotency via static source inspection (mirrors the
PR's own test_gemma4_lora_patch_preserves_signature_for_inspect)."""
import sys, types
import pytest

# sys.path: unsloth_zoo is pip-installed in CI


def test_lora_patch_uses_idempotency_guard_attrs():
    """Source-level check: the function gates rewrap via _unsloth_gemma4_patch."""
    import inspect
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.patch_gemma4_vllm_lora_support)
    assert '_unsloth_gemma4_class_patched' in src
    assert '_unsloth_gemma4_patch' in src
    # Guard before rewrapping supports_lora
    assert 'hasattr(original_supports_lora, "_unsloth_gemma4_patch")' in src
    # Guard before rewrapping create_lora_manager
    assert 'hasattr(vllm_lora_model_manager.create_lora_manager, "_unsloth_gemma4_patch")' in src


def test_k_eq_v_patch_uses_idempotency_guard_attrs():
    import inspect
    from unsloth_zoo import empty_model
    src = inspect.getsource(empty_model.patch_gemma4_vllm_k_eq_v_support)
    assert '_unsloth_gemma4_k_eq_v_patch' in src
    assert 'hasattr(stack_quantization_states, "_unsloth_gemma4_k_eq_v_patch")' in src
    # Early-return guard when private API is absent
    assert 'if stack_quantization_states is None' in src


def _install_vllm_stubs():
    """Install minimal vLLM stub modules so the patch functions can run."""
    # vllm root package needs to be a package (has __path__) for submodule imports
    if "vllm" not in sys.modules or not hasattr(sys.modules["vllm"], "__path__"):
        vllm_pkg = types.ModuleType("vllm")
        vllm_pkg.__path__ = []
        sys.modules["vllm"] = vllm_pkg

    # vllm.config (required transitively by unsloth_zoo.vllm_lora_worker_manager)
    cfg_mod = types.ModuleType("vllm.config")
    class _LoRAConfig:
        pass
    cfg_mod.LoRAConfig = _LoRAConfig
    sys.modules["vllm.config"] = cfg_mod

    # vllm.logger (also required transitively)
    logger_mod = types.ModuleType("vllm.logger")
    import logging
    def init_logger(name):
        return logging.getLogger(name)
    logger_mod.init_logger = init_logger
    sys.modules["vllm.logger"] = logger_mod

    # vllm.lora.layers (transitively required by vllm_lora_worker_manager)
    lora_layers_mod = types.ModuleType("vllm.lora.layers")
    class _LoRAMapping:
        pass
    lora_layers_mod.LoRAMapping = _LoRAMapping
    sys.modules.setdefault("vllm.lora", types.ModuleType("vllm.lora"))
    sys.modules["vllm.lora"].__path__ = []
    sys.modules["vllm.lora.layers"] = lora_layers_mod

    # vllm.lora.lora_model (transitively required)
    lora_model_mod = types.ModuleType("vllm.lora.lora_model")
    class _LoRAModel:
        pass
    lora_model_mod.LoRAModel = _LoRAModel
    sys.modules["vllm.lora.lora_model"] = lora_model_mod

    # vllm.lora.request
    lora_req_mod = types.ModuleType("vllm.lora.request")
    class _LoRARequest:
        pass
    lora_req_mod.LoRARequest = _LoRARequest
    sys.modules["vllm.lora.request"] = lora_req_mod

    # vllm.lora.worker_manager
    lora_wm_mod = types.ModuleType("vllm.lora.worker_manager")
    class _WorkerLoRAManager:
        pass
    lora_wm_mod.WorkerLoRAManager = _WorkerLoRAManager
    class _LRUCacheWorkerLoRAManager:
        pass
    lora_wm_mod.LRUCacheWorkerLoRAManager = _LRUCacheWorkerLoRAManager
    sys.modules["vllm.lora.worker_manager"] = lora_wm_mod

    # vllm.model_executor.models.interfaces
    interfaces_mod = types.ModuleType("vllm.model_executor.models.interfaces")
    def _orig_supports_lora(model):
        return False
    interfaces_mod.supports_lora = _orig_supports_lora
    sys.modules.setdefault("vllm.model_executor", types.ModuleType("vllm.model_executor"))
    sys.modules.setdefault("vllm.model_executor.models", types.ModuleType("vllm.model_executor.models"))
    sys.modules["vllm.model_executor.models.interfaces"] = interfaces_mod

    # vllm.lora.model_manager
    lora_mod = types.ModuleType("vllm.lora.model_manager")
    class _LoRAModelManager:
        pass
    lora_mod.LoRAModelManager = _LoRAModelManager
    def _orig_create(model, *args, **kwargs):
        return ("orig", model, args, kwargs)
    lora_mod.create_lora_manager = _orig_create
    sys.modules.setdefault("vllm.lora", types.ModuleType("vllm.lora"))
    sys.modules["vllm.lora.model_manager"] = lora_mod

    # vllm.v1.worker.lora_model_runner_mixin
    mixin_mod = types.ModuleType("vllm.v1.worker.lora_model_runner_mixin")
    mixin_mod.supports_lora = _orig_supports_lora
    sys.modules.setdefault("vllm.v1", types.ModuleType("vllm.v1"))
    sys.modules.setdefault("vllm.v1.worker", types.ModuleType("vllm.v1.worker"))
    sys.modules["vllm.v1.worker.lora_model_runner_mixin"] = mixin_mod

    # vllm.model_executor.models.gemma4_mm and gemma4 (stub classes)
    gemma4_mm = types.ModuleType("vllm.model_executor.models.gemma4_mm")
    class _Gemma4ForConditionalGeneration:
        pass
    gemma4_mm.Gemma4ForConditionalGeneration = _Gemma4ForConditionalGeneration
    sys.modules["vllm.model_executor.models.gemma4_mm"] = gemma4_mm
    gemma4 = types.ModuleType("vllm.model_executor.models.gemma4")
    class _Gemma4ForCausalLM:
        pass
    gemma4.Gemma4ForCausalLM = _Gemma4ForCausalLM
    sys.modules["vllm.model_executor.models.gemma4"] = gemma4

    return interfaces_mod, lora_mod, mixin_mod, _Gemma4ForConditionalGeneration, _Gemma4ForCausalLM


def test_patch_gemma4_vllm_lora_support_idempotent():
    pytest.importorskip("vllm.lora.models", reason="needs real vllm to run end-to-end; idempotency guarded by attrs (see test_lora_patch_uses_idempotency_guard_attrs)")
    interfaces_mod, lora_mod, mixin_mod, GM_VLM, GM_LM = _install_vllm_stubs()
    # Force re-import of empty_model so it sees fresh stubs
    if "unsloth_zoo.empty_model" in sys.modules:
        del sys.modules["unsloth_zoo.empty_model"]
    from unsloth_zoo import empty_model

    # First call patches
    empty_model.patch_gemma4_vllm_lora_support()
    first_supports_lora = interfaces_mod.supports_lora
    first_create_lora = lora_mod.create_lora_manager
    assert getattr(GM_VLM, "_unsloth_gemma4_class_patched", False)
    assert getattr(GM_LM, "_unsloth_gemma4_class_patched", False)
    assert getattr(first_supports_lora, "_unsloth_gemma4_patch", False)
    assert getattr(first_create_lora, "_unsloth_gemma4_patch", False)

    # Second + third call should be no-ops (no double-wrapping)
    empty_model.patch_gemma4_vllm_lora_support()
    empty_model.patch_gemma4_vllm_lora_support()
    assert interfaces_mod.supports_lora is first_supports_lora, "supports_lora got rewrapped"
    assert lora_mod.create_lora_manager is first_create_lora, "create_lora_manager got rewrapped"


def test_patched_supports_lora_returns_true_only_for_gemma4_classes():
    pytest.importorskip("vllm.lora.models", reason="needs real vllm to run end-to-end")
    interfaces_mod, lora_mod, mixin_mod, GM_VLM, GM_LM = _install_vllm_stubs()
    if "unsloth_zoo.empty_model" in sys.modules:
        del sys.modules["unsloth_zoo.empty_model"]
    from unsloth_zoo import empty_model
    empty_model.patch_gemma4_vllm_lora_support()

    # A Gemma4 class instance returns True
    gemma4_inst = GM_VLM()
    assert interfaces_mod.supports_lora(gemma4_inst) is True

    # A non-Gemma4 class falls through to the original (returns False here)
    class _Other:
        pass
    assert interfaces_mod.supports_lora(_Other()) is False


def test_patch_gemma4_vllm_k_eq_v_support_idempotent():
    # Install a BitsAndBytesModelLoader stub
    if "vllm.model_executor.model_loader.bitsandbytes_loader" in sys.modules:
        del sys.modules["vllm.model_executor.model_loader.bitsandbytes_loader"]
    bnb_mod = types.ModuleType("vllm.model_executor.model_loader.bitsandbytes_loader")
    sys.modules.setdefault("vllm.model_executor.model_loader",
                            types.ModuleType("vllm.model_executor.model_loader"))

    class _Loader:
        @staticmethod
        def _stack_quantization_states(self, model, qsd):
            return dict(qsd)

    bnb_mod.BitsAndBytesModelLoader = _Loader
    sys.modules["vllm.model_executor.model_loader.bitsandbytes_loader"] = bnb_mod

    if "unsloth_zoo.empty_model" in sys.modules:
        del sys.modules["unsloth_zoo.empty_model"]
    from unsloth_zoo import empty_model

    empty_model.patch_gemma4_vllm_k_eq_v_support()
    first = _Loader._stack_quantization_states
    assert getattr(first, "_unsloth_gemma4_k_eq_v_patch", False)

    empty_model.patch_gemma4_vllm_k_eq_v_support()
    empty_model.patch_gemma4_vllm_k_eq_v_support()
    assert _Loader._stack_quantization_states is first, "k_eq_v patch got re-wrapped"


def test_patch_gemma4_vllm_k_eq_v_support_noop_when_attr_missing():
    """If vLLM's loader doesn't expose _stack_quantization_states, patch is a no-op."""
    if "vllm.model_executor.model_loader.bitsandbytes_loader" in sys.modules:
        del sys.modules["vllm.model_executor.model_loader.bitsandbytes_loader"]
    bnb_mod = types.ModuleType("vllm.model_executor.model_loader.bitsandbytes_loader")
    sys.modules.setdefault("vllm.model_executor.model_loader",
                            types.ModuleType("vllm.model_executor.model_loader"))

    class _OldLoader:  # no _stack_quantization_states
        pass

    bnb_mod.BitsAndBytesModelLoader = _OldLoader
    sys.modules["vllm.model_executor.model_loader.bitsandbytes_loader"] = bnb_mod

    if "unsloth_zoo.empty_model" in sys.modules:
        del sys.modules["unsloth_zoo.empty_model"]
    from unsloth_zoo import empty_model

    # Should not raise
    empty_model.patch_gemma4_vllm_k_eq_v_support()
    # Should not have created the attribute either
    assert not hasattr(_OldLoader, "_stack_quantization_states")
