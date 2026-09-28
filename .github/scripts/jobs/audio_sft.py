"""Audio (ASR) LoRA SFT: Gemma4_(E2B)-Audio notebook (FastModel, its LoRA targets, UnslothVisionDataCollator)
on the first rows of kadirnar/Emilia-DE-B000000 (one ~7 MB parquet row group, range-read). Checks
input_features reached the model.

    python jobs/audio_sft.py --max-steps 3        # unsloth/gemma-4-E2B-it
    python jobs/audio_sft.py --synthetic          # offline: 16 kHz tones labelled "Ton N" (not speech)
    python jobs/audio_sft.py --tiny               # tiny-Gemma4 has no audio tower: exits 3 (skip)

Decoding the dataset's mp3 needs `soundfile`; without it the job falls back to --synthetic and
records data_source, so compare.py never pairs speech with tones silently.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as c  # noqa: E402
import _presets as P  # noqa: E402

pre = P.AUDIO_SFT["gemma4"]
p = c.base_parser("audio_sft", pre["model"], pre["tiny"], default_steps=3)
p.add_argument("--lora-r", type=int, default=pre["lora_r"])
p.add_argument("--rows", type=int, default=8)
p.add_argument("--max-seconds", type=float, default=6.0, help="clip each utterance to this length")
p.add_argument("--synthetic", action="store_true", help="tones instead of Emilia speech (offline)")
p.set_defaults(lr=pre["lr"], max_seq_length=pre["max_seq_length"])
a = c.resolve_args(p)

SR = 16000
EMILIA = "datasets/kadirnar/Emilia-DE-B000000/data/train-00000-of-00002.parquet"
SYSTEM = "You are an assistant that transcribes speech accurately."


def resample(x, sr_in, sr_out=SR):
    import numpy as np
    if sr_in == sr_out:
        return x.astype("float32")
    n = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype("float32")


def emilia_rows(n, max_s):
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open(EMILIA, block_size=4 << 20) as f:
        tb = pq.ParquetFile(f).read_row_group(0, columns=["text", "audio"])
    out = []
    for r in tb.to_pylist():
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        x = x.mean(axis=1) if x.ndim > 1 else x
        out.append((resample(x, sr)[: int(max_s * SR)], r["text"].strip()))
        if len(out) >= n:
            break
    return out


def synthetic_rows(n, max_s):
    import numpy as np
    t = np.arange(int(min(max_s, 2.0) * SR)) / SR
    return [((0.3 * np.sin(2 * np.pi * (220 + 55 * i) * t)).astype("float32"), f"Ton {i}.") for i in range(n)]


def to_messages(pairs):
    return [{"messages": [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": "Please transcribe this audio."}]},
        {"role": "assistant", "content": [{"type": "text", "text": y}]}]} for x, y in pairs]


class AudioKwargsProcessor:
    """Forwards to the processor; adds audio_kwargs max_length (30 s) when audio is passed. Some Gemma 4
    processors (tiny-Gemma4) default the feature extractor to truncation=True with max_length=None."""

    def __init__(self, proc, max_samples=30 * SR):
        self._proc, self._max = proc, max_samples

    def __call__(self, *args, **kw):
        if kw.get("audio") is not None:
            kw.setdefault("audio_kwargs", {}).setdefault("max_length", self._max)
        return self._proc(*args, **kw)

    def __getattr__(self, name):
        return getattr(self._proc, name)


class CountingCollator:
    """Counts batches carrying non-empty input_features; lists back to float32 arrays (Arrow round-trip)."""

    def __init__(self, inner):
        self.inner, self.batches, self.with_audio = inner, 0, 0

    def __call__(self, batch):
        import numpy as np
        fixed = []
        for b in batch:
            msgs = [{**m, "content": [{kk: (np.asarray(v, dtype="float32") if kk == "audio" else v)
                                       for kk, v in x.items() if v is not None} for x in m["content"]]}
                    for m in b["messages"]]
            fixed.append({**b, "messages": msgs})
        out = self.inner(fixed)
        self.batches += 1
        f = out.get("input_features") if hasattr(out, "get") else None
        self.with_audio += int(f is not None and f.numel() > 0)
        return out


with c.JobRecorder(a) as rec:
    c.seed_everything(a.seed)
    if a.backend != "hf":
        import unsloth  # noqa: F401
    import _compat as k
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    rec.set_backend(k.backend_name(a, c.detect_device()))
    source = "synthetic" if (a.synthetic or a.tiny) else "emilia"
    if source == "emilia":
        try:
            pairs = emilia_rows(a.rows, a.max_seconds)
        except ImportError as e:  # soundfile / pyarrow missing
            source, pairs = f"synthetic (fallback: {e})", synthetic_rows(a.rows, a.max_seconds)
    else:
        pairs = synthetic_rows(a.rows, a.max_seconds)
    rec.summary(data_source=source, rows=len(pairs), audio_seconds=[round(len(x) / SR, 2) for x, _ in pairs])

    if a.backend == "hf":
        rec.skip("audio_sft covers the Unsloth FastModel path only (--backend unsloth, or auto on an accelerator)")
    from unsloth import FastModel
    from unsloth.trainer import UnslothVisionDataCollator
    model, proc = FastModel.from_pretrained(model_name=a.model, max_seq_length=a.max_seq_length,
                                            load_in_4bit=False, dtype=None)
    if not any("audio_tower" in n for n, _ in model.named_modules()):
        rec.skip(f"{a.model} has no audio tower (audio_config null)")
    model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                     finetune_attention_modules=True, finetune_mlp_modules=True,
                                     r=a.lora_r, lora_alpha=pre["lora_alpha"], lora_dropout=0, bias="none",
                                     random_state=a.seed, target_modules=pre["target_modules"])
    collator = CountingCollator(UnslothVisionDataCollator(model, AudioKwargsProcessor(proc)))

    dropped = []
    ta = k.common_train_kwargs(a, os.path.join(os.path.dirname(a.out) or ".", "audio_run"))
    ta.update(per_device_train_batch_size=pre["batch"], optim="adamw_8bit", dataset_num_proc=1)
    cfg = k.make(SFTConfig, dropped, remove_unused_columns=False, dataset_text_field="",
                 dataset_kwargs={"skip_prepare_dataset": True}, max_length=a.max_seq_length, **ta)
    ds = Dataset.from_list([{"messages": [{**m, "content": [
        {kk: (v.tolist() if kk == "audio" else v) for kk, v in x.items()} for x in m["content"]]} for m in r["messages"]]}
        for r in to_messages(pairs)])
    trainer = SFTTrainer(model=model, train_dataset=ds, data_collator=collator, args=cfg,
                         callbacks=[c.metrics_callback(rec)], **k.tokenizer_kwarg(SFTTrainer, proc))
    rec.summary(dropped_config_args=dropped, notebook=pre["notebook"],
                trainable_params=sum(x.numel() for x in model.parameters() if x.requires_grad))

    before = c.adapter_fingerprint(model)
    out = trainer.train()
    after = c.adapter_fingerprint(model)
    rec.summary(train_runtime_s=out.metrics.get("train_runtime"), final_loss=out.training_loss,
                batches=collator.batches, batches_with_audio=collator.with_audio)

    rec.standard_training_checks()
    rec.backend_check()
    rec.check("adapter_changed", *c.adapter_changed(before, after))
    rec.check("audio_reached_model", collator.batches > 0 and collator.with_audio == collator.batches,
              f"{collator.with_audio}/{collator.batches} batches carried input_features")
