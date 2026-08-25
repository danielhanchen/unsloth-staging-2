"""Drive image generation (e.g. Nano Banana) and save the decoded PNG.

    GEMINI_API_KEY=... python -m studio_test_kit.examples.image_gen \
        --port 8902 --password 'YourBootstrap!' \
        --model gemini-2.5-flash-image \
        --prompt 'A red panda eating ramen in the rain, neon lighting'
"""

import argparse
import asyncio
import os
from pathlib import Path

from studio_test_kit.auth import gemini_provider, login, seed_init_script
from studio_test_kit.flows import image_generation
from studio_test_kit.ui import open_chat


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8902)
    ap.add_argument("--password", required=True)
    ap.add_argument("--username", default="unsloth")
    ap.add_argument("--model", default="gemini-2.5-flash-image")
    ap.add_argument("--prompt", default="A red panda eating ramen in the rain")
    ap.add_argument("--out", type=Path, default=Path("outputs/image_gen"))
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    auth = await login(base, args.username, args.password)
    providers = [
        gemini_provider(api_key=os.environ["GEMINI_API_KEY"], models=[args.model])
    ]
    init = seed_init_script(auth, providers)

    async with open_chat(
        base, init_scripts=[init],
        video_dir=args.out / "video",
        video_name="image_gen",
        transcode_mp4=True,
        headless=True,
    ) as sp:
        r = await image_generation(sp, args.model, args.prompt, args.out)
    r.attach_video(sp)
    print(f"Image saved to {r.artefacts['image_path']} "
          f"({r.artefacts['image_bytes']} bytes)")
    if r.video_webm:
        print(f"  video webm: {r.video_webm}")
    if r.video_mp4:
        print(f"  video mp4:  {r.video_mp4}")


if __name__ == "__main__":
    asyncio.run(main())
