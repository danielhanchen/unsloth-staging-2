"""Measure and photograph the Settings search offsets at normal and larger spacing."""

from pathlib import Path

from studio_test_kit.auth import seed_init_script
from studio_test_kit.ui import open_chat


async def drive(session, out_dir: Path, label: str, **_):
    init = seed_init_script(
        type(
            "Auth",
            (),
            {"access_token": session.access_token, "refresh_token": session.refresh_token},
        )(),
        [],
    )
    async with open_chat(
        session.base_url, init_scripts = [init], viewport = (1440, 1000), headless = True
    ) as sp:
        page = sp.page
        await page.get_by_role("button", name = "Settings", exact = True).click()
        field = page.locator('input[data-type-to-activate="settings-search"]')
        await field.wait_for(state = "visible")
        await field.fill("theme")
        await field.locator("..").locator("button").wait_for(state = "visible")
        facts = {}
        shots = []
        for name, scale in [("default", 1), ("large", 1.4)]:
            await page.evaluate(
                '(s) => document.documentElement.style.setProperty("--ui-space-scale", String(s))',
                scale,
            )
            await page.evaluate("""async () => {
                await Promise.all(document.getAnimations()
                    .filter(a => Number.isFinite(a.effect?.getTiming().iterations))
                    .map(a => a.finished.catch(() => {})));
            }""")
            values = await field.evaluate("""el => {
                const box = el.parentElement;
                const icon = box.querySelector('svg');
                const clear = box.querySelector('button');
                if (!icon || !clear) throw Error('Search controls missing');
                return {
                    input_height: parseFloat(getComputedStyle(el).height),
                    text_inset: parseFloat(getComputedStyle(el).paddingLeft),
                    icon_left: parseFloat(getComputedStyle(icon).left),
                    clear_right: parseFloat(getComputedStyle(clear).right)
                };
            }""")
            assert values["input_height"] > 0, values
            facts.update({f"{name}_{k}": v for k, v in values.items()})
            box = await field.locator("..").bounding_box()
            assert box
            shot = out_dir / f"{label.lower()}_settings_{name}.png"
            await page.screenshot(path = str(shot), clip = box, animations = "disabled")
            if name == "large":
                shots.append(shot)
        return shots, facts
