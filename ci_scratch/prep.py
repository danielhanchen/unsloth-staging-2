"""Wire the review-only proxy e2e probe into the Studio crate (staging CI only)."""
import pathlib
import shutil
import sys

root = pathlib.Path(__file__).resolve().parent.parent
src = root / "studio" / "src-tauri" / "src"
shutil.copy(root / "ci_scratch" / "proxy_e2e_probe.rs", src / "proxy_e2e_probe.rs")

main = src / "main.rs"
text = main.read_text()
if "mod proxy_e2e_probe;" not in text:
    if "mod process;\n" not in text:
        sys.exit("could not find `mod process;` in main.rs")
    text = text.replace("mod process;\n", "mod process;\nmod proxy_e2e_probe;\n", 1)
    main.write_text(text)
print("probe wired in")
