"""Apply one deliberate break to delete-thread-message.ts, from $FIND / $REPL."""

import os
import pathlib

p = pathlib.Path("src/features/chat/utils/delete-thread-message.ts")
text = p.read_text()
find, repl = os.environ["FIND"], os.environ["REPL"]
if find not in text:
    raise SystemExit(f"mutation anchor missing: {find!r}")
p.write_text(text.replace(find, repl, 1))
print(f"mutated: {find!r} -> {repl!r}")
