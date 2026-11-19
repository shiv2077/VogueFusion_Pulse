"""Drive the web app in headless Chrome against the running stack.

Generate (one reference, workflow v1) -> "Cost this design" -> Colors (extract)
-> Dye Recipes -> Cost (calculate) -> Status. Fails on any page error or on a
view that never shows its expected content. Screenshots of each view go to
$VOGUEFUSION_DATA/ui_check/.

Needs Playwright (pip) and a system Chrome; exits 2 if either is missing.
Run: PYTHONPATH=src python scripts/ui_check.py
"""

from __future__ import annotations

import sys
import time

from voguefusion import config

REF = "dataset/initial_dataset/Saree/image_1.jpg"
WAIT_MS = 180_000


def main() -> int:
    try:
        from playwright.sync_api import Error, sync_playwright
    except ImportError:
        print("playwright not installed", file=sys.stderr)
        return 2
    cfg = config.load()
    base = f"http://{cfg['api']['host']}:{cfg['api']['port']}"
    out = config.data_dir(cfg) / "ui_check"
    out.mkdir(exist_ok=True)
    problems: list[str] = []

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=True)
        except Error as e:
            print(f"Chrome not available: {e}", file=sys.stderr)
            return 2
        pg = browser.new_page(viewport={"width": 1915, "height": 1000})
        pg.on("pageerror", lambda e: problems.append(f"page error: {e}"))

        def expect(selector: str, where: str) -> bool:
            try:
                pg.locator(selector).first.wait_for(timeout=WAIT_MS)
                return True
            except Error:
                problems.append(f"{where}: {selector} never appeared")
                return False

        pg.goto(f"{base}/#generate")
        pg.wait_for_selector("#refs .drop")
        pg.select_option("#workflow", "v1")
        with pg.expect_file_chooser() as chooser:
            pg.locator("#refs .drop").first.click()
        chooser.value.set_files(str(config.REPO_ROOT / REF))
        pg.wait_for_function(
            "document.querySelector('#refs .preview') && !document.querySelector('#refs .status')"
        )
        t0 = time.monotonic()
        pg.click("#generate")
        if not expect("#gen-out .grid img", "Generate"):
            browser.close()
            print("\n".join(problems))
            return 1
        pg.wait_for_function(
            "[...document.querySelectorAll('#gen-out .grid img')]"
            ".every(i => i.complete && i.naturalWidth > 0)"
        )
        print(f"generate: {time.monotonic() - t0:.1f} s")
        pg.screenshot(path=str(out / "generate.png"))

        pg.locator("#gen-out .card").first.hover()
        pg.locator("[data-cost]").first.click()
        pg.click("#extract")
        expect("#colors-out .chart", "Colors")
        pg.screenshot(path=str(out / "colors.png"))

        pg.click("nav a[data-view=dyes]")
        expect("#dyes-out .recipe", "Dye Recipes")
        pg.screenshot(path=str(out / "dyes.png"))

        pg.click("nav a[data-view=cost]")
        if not pg.locator("#cost-out .total").count():
            pg.click("#calc")
        expect("#cost-out .total", "Cost")
        pg.screenshot(path=str(out / "cost.png"))

        pg.click("nav a[data-view=status]")
        expect("#status-out .stats", "Status")
        pg.screenshot(path=str(out / "status.png"))
        browser.close()

    print("\n".join(problems) or f"ui check: ok (screenshots in {out})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
