# ARM64 base-vs-head (KleidiAI) sd-cli A/B: interleaved rounds, base twice per round (A/A), stage timings from the log,
# per-(type, buffer) weight counts from the SD_BUFT_STATS instrumentation, PSNR vs the first base image.
import json, os, re, subprocess, sys, time
import numpy as np
from PIL import Image

OUT = sys.argv[1]; DIT = sys.argv[2]; TAG = sys.argv[3]
LLM, VAE = os.environ['LLM'], os.environ['VAE']
PX, STEPS, ROUNDS, T = int(os.environ.get('PX', 256)), int(os.environ.get('STEPS', 4)), int(os.environ.get('ROUNDS', 3)), int(os.environ.get('THREADS', os.cpu_count()))
ARMS = [('base', 'bin-base'), ('base_aa', 'bin-base'), ('head', 'bin-head')]
os.makedirs(OUT, exist_ok=True)
res = open(os.path.join(OUT, 'results.jsonl'), 'a')
PAT = {'cond_s': r'get_learned_condition completed, taking ([\d.]+)s', 'sample_s': r'sampling completed, taking ([\d.]+)s',
       'decode_s': r'decode_first_stage completed, taking ([\d.]+)s', 'gen_s': r'generate_image completed in ([\d.]+)s',
       'load_s': r'loading tensors completed, taking ([\d.]+)s'}
for rnd in range(1, ROUNDS + 1):
    k = (rnd - 1) % len(ARMS)
    for arm, bindir in ARMS[k:] + ARMS[:k]:
        png = os.path.join(OUT, f'{TAG}_{arm}_r{rnd}.png'); log = png[:-4] + '.log'
        cmd = [os.path.join(bindir, 'sd-cli'), '--diffusion-model', DIT, '--llm', LLM, '--vae', VAE, '-p', 'a photo of a red fox sitting in fresh snow, golden hour, detailed fur',
               '-W', str(PX), '-H', str(PX), '--steps', str(STEPS), '--cfg-scale', '1', '--seed', '42', '-t', str(T), '-o', png, '-v']
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, 'SD_BUFT_STATS': '1'})
        wall = time.time() - t0
        txt = p.stdout + p.stderr
        open(log, 'w').write(txt)
        rec = {'tag': TAG, 'arm': arm, 'round': rnd, 'rc': p.returncode, 'wall_s': round(wall, 2)}
        for key, pat in PAT.items():
            m = re.findall(pat, txt)
            rec[key] = float(m[-1]) if m else None
        stats = {}
        for ty, bt in re.findall(r'buft-stat (\S+) (\S+)', txt):
            stats[f'{ty}:{bt}'] = stats.get(f'{ty}:{bt}', 0) + 1
        rec['buft'] = stats
        rec['kleidi_lines'] = [l.strip()[-120:] for l in txt.splitlines() if 'kleidiai' in l.lower()][:8]
        rec['cpu_variant'] = [l.strip()[-120:] for l in txt.splitlines() if re.search(r'load_backend|ggml-cpu-|CPU : ', l)][:4]
        if p.returncode == 0:
            ref = os.path.join(OUT, f'{TAG}_base_r1.png')
            if os.path.exists(ref):
                a = np.asarray(Image.open(ref).convert('RGB')).astype(np.float64); b = np.asarray(Image.open(png).convert('RGB')).astype(np.float64)
                mse = float(((a - b) ** 2).mean())
                rec['psnr_vs_base_r1'] = 'inf' if mse == 0 else round(10 * np.log10(255 ** 2 / mse), 2)
                rec['maxabs_vs_base_r1'] = int(np.abs(a - b).max())
        else:
            rec['tail'] = txt[-1500:]
        print(json.dumps(rec), flush=True); res.write(json.dumps(rec) + '\n'); res.flush()
