"""Exercise the Search and Code composer pills."""

import argparse
import asyncio
import os
from pathlib import Path

from studio_test_kit.auth import gemini_provider, login, seed_init_script
from studio_test_kit.flows import tool_pills
from studio_test_kit.ui import open_chat


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8902)
    ap.add_argument("--password", required=True)
    ap.add_argument("--username", default="unsloth")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--out", type=Path, default=Path("outputs/tools_pills"))
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
        video_name="tools_pills",
        transcode_mp4=True,
        headless=True,
    ) as sp:
        r = await tool_pills(sp, args.model, args.out)
    r.attach_video(sp)
    print(f"Wrote {len(r.screenshots)} screenshots to {r.out_dir}")
    if r.video_webm:
        print(f"  video webm: {r.video_webm}")
    if r.video_mp4:
        print(f"  video mp4:  {r.video_mp4}")


if __name__ == "__main__":
    asyncio.run(main())
