"""Pre-PR vs post-PR side-by-side: install both, drive the same chat flow
on both, then compose a hstack video + an hstack per-turn screenshot.

This is the canonical "screenshot the behavior change" workflow.

    GEMINI_API_KEY=... python -m studio_test_kit.examples.side_by_side \
        --pre-branch main \
        --post-branch feat/gemini-provider \
        --pre-port 8901 \
        --post-port 8902 \
        --model gemini-2.5-flash
"""

import argparse
import asyncio
import os
from pathlib import Path

from studio_test_kit.auth import gemini_provider, login, seed_init_script
from studio_test_kit.compose import hstack_images, hstack_videos
from studio_test_kit.flows import multi_turn_chat
from studio_test_kit.lifecycle import install_studio, launch_studio
from studio_test_kit.ui import open_chat


PROMPTS = [
    "Translate 'good morning' into Japanese.",
    "Now in a pirate voice.",
    "Summarize this thread in 5 words.",
]


async def drive(
    port: int, password: str, model: str, out_dir: Path, video_dir: Path,
    video_name: str,
) -> tuple[list[Path], Path]:
    base = f"http://127.0.0.1:{port}"
    auth = await login(base, "unsloth", password)
    providers = [
        gemini_provider(api_key=os.environ["GEMINI_API_KEY"], models=[model])
    ]
    init = seed_init_script(auth, providers)
    async with open_chat(
        base, init_scripts=[init], video_dir=video_dir,
        video_name=video_name, headless=True,
    ) as sp:
        r = await multi_turn_chat(sp, model, PROMPTS, out_dir)
    r.attach_video(sp)
    assert r.video_webm is not None, "video was not recorded"
    return r.screenshots, r.video_webm


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-branch", default="main")
    ap.add_argument("--post-branch", required=True)
    ap.add_argument("--pre-port", type=int, default=8901)
    ap.add_argument("--post-port", type=int, default=8902)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--root", type=Path, default=Path("outputs/side_by_side"))
    ap.add_argument("--studio-home-root", type=Path,
                    default=Path("temp/side_by_side_studios"))
    args = ap.parse_args()

    pre = install_studio(args.pre_branch, args.studio_home_root / "pre")
    post = install_studio(args.post_branch, args.studio_home_root / "post")
    launch_studio(pre, args.pre_port, args.root / "pre.log")
    launch_studio(post, args.post_port, args.root / "post.log")

    pre_shots, pre_video = await drive(
        args.pre_port, pre.bootstrap_password or "", args.model,
        args.root / "pre", args.root / "pre" / "video", "pre",
    )
    post_shots, post_video = await drive(
        args.post_port, post.bootstrap_password or "", args.model,
        args.root / "post", args.root / "post" / "video", "post",
    )

    combined = args.root / "combined"
    combined.mkdir(parents=True, exist_ok=True)
    for i, (left, right) in enumerate(zip(pre_shots, post_shots), start=1):
        hstack_images(left, right, combined / f"sxs_{i:02d}.png",
                      label_left=args.pre_branch, label_right=args.post_branch)

    sxs_mp4 = combined / "sxs.mp4"
    hstack_videos(pre_video, post_video, sxs_mp4)
    print(f"Wrote combined comparison to {combined}")
    print(f"  pre webm:  {pre_video}")
    print(f"  post webm: {post_video}")
    print(f"  sxs mp4:   {sxs_mp4}")


if __name__ == "__main__":
    asyncio.run(main())
