"""Hash clean_text's keep/drop decision for every code point.

The value must be identical on every OS, architecture and Python version, or the
same corpus cleans differently depending on where training runs.
"""
import hashlib, importlib.util, os, platform, sys, types

sys.modules.setdefault("datasets", types.ModuleType("datasets"))
sys.modules["datasets"].Dataset = type("Dataset", (), {"from_dict": staticmethod(lambda d: d)})
spec = importlib.util.spec_from_file_location(
    "raw_text", os.path.join("unsloth", "dataprep", "raw_text.py")
)
m = importlib.util.module_from_spec(spec)
sys.modules["raw_text"] = m
spec.loader.exec_module(m)

table = m.TextPreprocessor._TEXT_CHARS
digest = hashlib.sha256()
for cp in range(0x110000):
    digest.update(b"1" if table[cp] is not None else b"0")
got = digest.hexdigest()

EXPECTED = "2f22530d105fe2d153be3c0d153ea1df1260e217822f3ce862aad8580e2970f5"
import unicodedata
print(f"{platform.system()} {platform.machine()} py{sys.version.split()[0]} "
      f"ucd{unicodedata.unidata_version} -> {got}")
sample = "Le café 机器学习 Привет ❤️"
assert m.TextPreprocessor().clean_text(sample) == sample, "non-ASCII text was altered"
assert got == EXPECTED, f"decision differs from the reference: {got} != {EXPECTED}"
print("OK: identical to the reference decision")
