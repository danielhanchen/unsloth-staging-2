"""Drive a 4-turn conversation against any external provider.

Run:
    GEMINI_API_KEY=... python -m studio_test_kit.examples.multi_turn \
        --port 8902 --password 'YourBootstrap!' --model gemini-2.5-flash

This assumes Studio is already running on `--port`. Use
`studio_test_kit.lifecycle.install_studio` / `launch_studio` to spin it
up programmatically from CI.
"""

import argparse
import asyncio
import os
from pathlib import Path

from studio_test_kit.auth import gemini_provider, login, seed_init_script
from studio_test_kit.flows import multi_turn_chat
from studio_test_kit.ui import open_chat


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8902)
    ap.add_argument("--password", required=True)
    ap.add_argument("--username", default="unsloth")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--out", type=Path, default=Path("outputs/multi_turn"))
    ap.add_argument("--headless", action="store_true", default=True)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    auth = await login(base, args.username, args.password)
    providers = [
        gemini_provider(api_key=os.environ["GEMINI_API_KEY"], models=[args.model])
    ]
    init = seed_init_script(auth, providers)

    prompts = [
        "Translate 'good morning' into Japanese.",
        "Now in a pirate voice.",
        "Summarize this thread in 5 words.",
        "And one emoji that captures it.",
    ]

    async with open_chat(
        base, init_scripts=[init],
        video_dir=args.out / "video",
        video_name="multi_turn",
        transcode_mp4=True,
        headless=args.headless,
    ) as sp:
        r = await multi_turn_chat(sp, args.model, prompts, args.out)
    r.attach_video(sp)
    print(f"Wrote {len(r.screenshots)} screenshots to {r.out_dir}")
    if r.video_webm:
        print(f"  video webm: {r.video_webm} ({r.video_webm.stat().st_size} B)")
    if r.video_mp4:
        print(f"  video mp4:  {r.video_mp4} ({r.video_mp4.stat().st_size} B)")


if __name__ == "__main__":
    asyncio.run(main())
