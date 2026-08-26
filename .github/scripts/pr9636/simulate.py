"""Exhaustive simulation of Studio's MCP tool-result flattening (PR #9636).

Every case is built from the REAL mcp / fastmcp objects a server actually produces and
run through the checkout's own ``_flatten_result``. Expectations are declared up front,
so a case cannot "pass" by documenting whatever the code happens to do.

    python simulate.py <checkout-root> [--verbose] [--only <substring>]

Exit 0 when every case matches, 1 otherwise. Designed to run identically on Linux,
macOS and Windows: no shell, no forward-slash assumptions, tmp paths via tempfile.
"""

import argparse
import base64
import json
import os
import re
import sys
import tempfile
from pathlib import Path

# --- checkout under test ------------------------------------------------------

def load_backend(checkout):
    backend = Path(checkout).resolve() / "studio" / "backend"
    if not backend.is_dir():
        sys.exit(f"not a checkout: {checkout}")
    sys.path.insert(0, str(backend))
    from core.inference import mcp_client
    return mcp_client


# --- fixtures -----------------------------------------------------------------

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_B64 = base64.b64encode(PNG_BYTES).decode()
JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 32).decode()

# a canonical media type: lowercase, no parameters, no whitespace. This is what may
# reach `data:${mimeType};base64,` in tool-fallback.tsx.
CANONICAL_MIME = re.compile(r"^image/[a-z0-9][a-z0-9.+-]*$")


class Case:
    __slots__ = ("id", "group", "result", "images", "body", "why")

    def __init__(self, id, group, result, images, body = None, why = ""):
        self.id = id
        self.group = group
        self.result = result
        self.images = images      # expected list of mimeTypes, in block order
        self.body = body          # expected visible text, or None to skip the check
        self.why = why


def build_cases(mt, FastResult, File, Image, AnyUrl, tmp):
    """All simulation cases. mt is mcp.types; FastResult is fastmcp's CallToolResult."""

    from types import SimpleNamespace

    def res(*blocks, structured = None, is_error = False):
        # fastmcp's CallToolResult is what client.call_tool returns, so it is what
        # _flatten_result is really handed (mcp_client.call_tool_sync).
        return FastResult(
            content = list(blocks), structured_content = structured,
            meta = None, is_error = is_error,
        )

    def text(t):
        return mt.TextContent(type = "text", text = t)

    def image(mime = "image/png", data = PNG_B64):
        return mt.ImageContent(type = "image", data = data, mimeType = mime)

    def audio(mime = "audio/wav", data = PNG_B64):
        return mt.AudioContent(type = "audio", data = data, mimeType = mime)

    def blob(mime = "image/png", uri = "file:///out/gen.png", data = PNG_B64):
        return mt.EmbeddedResource(
            type = "resource",
            resource = mt.BlobResourceContents(uri = AnyUrl(uri), mimeType = mime, blob = data),
        )

    def etext(t, uri = "file:///out/log.txt", mime = "text/plain"):
        return mt.EmbeddedResource(
            type = "resource",
            resource = mt.TextResourceContents(uri = AnyUrl(uri), mimeType = mime, text = t),
        )

    def rlink(uri = "file:///out/gen.png", name = "gen.png"):
        return mt.ResourceLink(type = "resource_link", uri = AnyUrl(uri), name = name)

    def fastmcp(helper):
        return helper.to_resource_content() if hasattr(helper, "to_resource_content") else helper.to_image_content()

    png_path = tmp / "gen.png"
    png_path.write_bytes(PNG_BYTES)
    pdf_path = tmp / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    svg_path = tmp / "chart.svg"
    svg_path.write_bytes(b"<svg xmlns='http://www.w3.org/2000/svg'/>")

    ATTACH1 = "[1 image attached; displayed to the user]"
    ATTACH2 = "[2 images attached; displayed to the user]"
    OMIT1 = "[1 image omitted (too large)]"
    C = []

    # -- 1. the reported bug: EmbeddedResource carries its payload on resource.blob ---
    for mime, want in [
        ("image/png", "image/png"),
        ("image/jpeg", "image/jpeg"),
        ("image/gif", "image/gif"),
        ("image/webp", "image/webp"),
        ("image/bmp", "image/bmp"),
        ("image/avif", "image/avif"),
        ("image/apng", "image/apng"),
        ("image/svg+xml", "image/svg+xml"),
        ("image/tiff", "image/tiff"),
        ("image/vnd.microsoft.icon", "image/vnd.microsoft.icon"),
        ("image/heic", "image/heic"),
    ]:
        C.append(Case(f"embedded {mime}", "embedded-image", res(blob(mime = mime)), [want], ATTACH1))

    # -- 2. media types are case-insensitive and may carry parameters (RFC 9110 8.3.1) --
    for mime, want in [
        ("IMAGE/PNG", "image/png"),
        ("Image/Jpeg", "image/jpeg"),
        ("image/PNG", "image/png"),
        ("  image/png  ", "image/png"),
        ("image/png; charset=binary", "image/png"),
        ("image/png;charset=binary", "image/png"),
        ("image/png ; q=1 ; charset=x", "image/png"),
        ("IMAGE/PNG; CHARSET=BINARY", "image/png"),
    ]:
        C.append(Case(f"embedded {mime!r}", "mime-normalisation", res(blob(mime = mime)), [want], ATTACH1))

    # -- 3. fastmcp File(data=..., format=X) labels the payload application/<format> ---
    for fmt, want in [
        ("png", "image/png"), ("jpeg", "image/jpeg"), ("jpg", "image/jpeg"),
        ("gif", "image/gif"), ("webp", "image/webp"), ("bmp", "image/bmp"),
        ("avif", "image/avif"), ("apng", "image/apng"), ("tif", "image/tiff"),
        ("tiff", "image/tiff"), ("ico", "image/vnd.microsoft.icon"),
        ("svg", "image/svg+xml"), ("svg+xml", "image/svg+xml"),
        ("heic", "image/heic"), ("heif", "image/heif"), ("jxl", "image/jxl"),
    ]:
        C.append(Case(f"embedded application/{fmt}", "fastmcp-alias",
                      res(blob(mime = f"application/{fmt}")), [want], ATTACH1))
        C.append(Case(f"embedded APPLICATION/{fmt.upper()}", "fastmcp-alias",
                      res(blob(mime = f"APPLICATION/{fmt.upper()}")), [want], ATTACH1))

    # -- 4. real fastmcp helpers, the objects a live server emits --------------------
    C += [
        Case("fastmcp File(path=gen.png)", "fastmcp-real", res(fastmcp(File(path = png_path))), ["image/png"], ATTACH1),
        Case("fastmcp File(path=chart.svg)", "fastmcp-real", res(fastmcp(File(path = svg_path))), ["image/svg+xml"], ATTACH1),
        Case("fastmcp File(data,format=png)", "fastmcp-real", res(fastmcp(File(data = PNG_BYTES, format = "png"))), ["image/png"], ATTACH1),
        Case("fastmcp File(data,format=webp)", "fastmcp-real", res(fastmcp(File(data = PNG_BYTES, format = "webp"))), ["image/webp"], ATTACH1),
        Case("fastmcp File(data,format=svg)", "fastmcp-real", res(fastmcp(File(data = PNG_BYTES, format = "svg"))), ["image/svg+xml"], ATTACH1),
        Case("fastmcp Image(data,format=png)", "fastmcp-real", res(fastmcp(Image(data = PNG_BYTES, format = "png"))), ["image/png"], ATTACH1),
        Case("fastmcp Image(path=gen.png)", "fastmcp-real", res(fastmcp(Image(path = png_path))), ["image/png"], ATTACH1),
        Case("fastmcp File(path=doc.pdf)", "fastmcp-real", res(fastmcp(File(path = pdf_path))), [], ""),
        Case("fastmcp File(data) no format", "fastmcp-real", res(fastmcp(File(data = PNG_BYTES))), [], ""),
    ]

    # -- 5. non-images stay ignored --------------------------------------------------
    for mime in ["application/pdf", "application/json", "application/octet-stream",
                 "application/zip", "application/xml", "text/html", "text/csv",
                 "audio/wav", "video/mp4", "font/woff2", "application/x-tar"]:
        C.append(Case(f"embedded {mime}", "non-image", res(blob(mime = mime)), [], ""))
    C += [
        Case("audio block", "non-image", res(audio()), [], ""),
        Case("audio block image mime", "non-image", res(audio(mime = "audio/mpeg")), [], ""),
    ]

    # -- 6. malformed / hostile media types must not reach the data url --------------
    for mime in ["image/", "image", "/png", "image//png", "", " ", "application/",
                 "application/exe", "application/x-msdownload", "image\n/png",
                 "image/png\nX-Injected: 1", "image/<script>", 'image/png"',
                 "application/png; charset=x", "*/*", "image/*"]:
        # the only assertion that matters: whatever survives must be canonical.
        C.append(Case(f"malformed mime {mime!r}", "malformed-mime", res(blob(mime = mime)), None, None,
                      why = "must be rejected or canonicalised, never passed through raw"))
    # pydantic rejects a non-string mimeType, so a conforming server cannot send one.
    # These are duck-typed blocks: defence in depth for a non-pydantic client only.
    for mime in [42, b"image/png", ["image/png"], {"mime": "image/png"}, True]:
        loose = SimpleNamespace(type = "resource", resource = SimpleNamespace(
            uri = "file:///out/gen.bin", mimeType = mime, blob = PNG_B64))
        C.append(Case(f"non-string mime {mime!r}", "malformed-mime", res(loose), [], "",
                      why = "non-string mime must not crash and must not render"))

    # -- 7. URI inference when mimeType is absent (BlobResourceContents.mimeType is optional)
    for uri, want in [
        ("file:///out/gen.png", ["image/png"]),
        ("file:///out/gen.PNG", ["image/png"]),
        ("file:///out/gen.jpeg", ["image/jpeg"]),
        ("file:///out/gen.svg", ["image/svg+xml"]),
        ("file:///out/gen.png?download=1", ["image/png"]),
        ("file:///out/gen.png#preview", ["image/png"]),
        ("file:///out/gen.png?a=1#b", ["image/png"]),
        ("https://h/out/gen.png?sig=abc123", ["image/png"]),
        ("file:///C:/out/gen.png", ["image/png"]),
        ("file://server/share/gen.png", ["image/png"]),
        ("file:///out/my%20image.png", ["image/png"]),
        ("file:///out/%E5%9B%BE%E7%89%87.png", ["image/png"]),
        ("file:///out/\u56fe\u7247.png", ["image/png"]),
        ("mcp://server/resources/gen.png", ["image/png"]),
        ("resource://images/gen.png", ["image/png"]),
        ("data:image/png;base64,iVBORw0KGgo=", ["image/png"]),
        # must NOT be inferred as an image
        ("file:///out/download?name=x.png", []),
        ("file:///out/download#x.png", []),
        ("file:///out/gen.pdf", []),
        ("data:application/pdf;base64,JVBERi0=", []),
        ("file:///out/gen", []),
        ("file:///out/", []),
        ("file:///out/gen%2Epng", []),
        ("resource://gen.png", []),
        ("file:///out/archive.tar.gz", []),
        ("file:///out/notes.txt", []),
    ]:
        C.append(Case(f"no mime, uri {uri}", "uri-inference",
                      res(blob(mime = None, uri = uri)), want, ATTACH1 if want else "",
                      why = "query and fragment are not part of the name"))

    # an explicit non-image mime must win over an image-looking URI
    C.append(Case("pdf mime, .png uri", "uri-inference",
                  res(blob(mime = "application/pdf", uri = "file:///out/gen.png")), [], "",
                  why = "declared type wins over the name"))
    C.append(Case("png mime, .pdf uri", "uri-inference",
                  res(blob(mime = "image/png", uri = "file:///out/doc.pdf")), ["image/png"], ATTACH1))

    # -- 8. embedded text resources --------------------------------------------------
    C += [
        Case("embedded text", "embedded-text", res(etext("saved to /out/gen.png")), [], "saved to /out/gen.png"),
        Case("embedded empty text", "embedded-text", res(etext("")), [], ""),
        Case("embedded whitespace text", "embedded-text", res(etext("   ")), [], "   "),
        Case("embedded multiline text", "embedded-text", res(etext("a\nb\nc")), [], "a\nb\nc"),
        Case("embedded unicode text", "embedded-text", res(etext("\u56fe\u7247 saved \u2713")), [], "\u56fe\u7247 saved \u2713"),
        Case("embedded json text", "embedded-text", res(etext('{"ok": true}', mime = "application/json")), [], '{"ok": true}'),
        Case("text + embedded text", "embedded-text", res(text("first"), etext("second")), [], "first\nsecond"),
    ]

    # -- 9. resource links -----------------------------------------------------------
    C += [
        Case("resource link named", "resource-link", res(rlink()), [], "[resource: gen.png <file:///out/gen.png>]"),
        # ResourceLink.name is required and non-null in the MCP schema, so the unnamed
        # branch of _block_link is only reachable from a non-conforming client.
        Case("resource link unnamed", "resource-link",
             res(SimpleNamespace(type = "resource_link", uri = "file:///out/gen.png", name = None)),
             [], "[resource: <file:///out/gen.png>]"),
        Case("resource link https", "resource-link", res(rlink(uri = "https://h/a.png", name = "a.png")), [],
             "[resource: a.png <https://h/a.png>]"),
        Case("text then link", "resource-link", res(text("hi"), rlink()), [],
             "hi\n[resource: gen.png <file:///out/gen.png>]"),
        Case("link then text", "resource-link", res(rlink(), text("hi")), [],
             "[resource: gen.png <file:///out/gen.png>]\nhi"),
    ]

    # -- 10. behaviour that must not change ------------------------------------------
    C += [
        Case("text only", "unchanged", res(text("hello")), [], "hello"),
        Case("empty text only", "unchanged", res(text("")), [], ""),
        Case("no content", "unchanged", res(), [], ""),
        Case("direct image", "unchanged", res(image()), ["image/png"], ATTACH1),
        Case("direct image jpeg", "unchanged", res(image(mime = "image/jpeg", data = JPEG_B64)), ["image/jpeg"], ATTACH1),
        Case("text + direct image", "unchanged", res(text("rendered"), image()), ["image/png"], f"rendered\n{ATTACH1}"),
        Case("two direct images", "unchanged", res(image(), image(mime = "image/jpeg", data = JPEG_B64)),
             ["image/png", "image/jpeg"], ATTACH2),
        Case("structured only", "unchanged", res(structured = {"a": 1}), [], "{'a': 1}"),
        Case("structured + text", "unchanged", res(text("hello"), structured = {"a": 1}), [], "hello"),
        Case("structured + image", "unchanged", res(image(), structured = {"a": 1}), ["image/png"], f"{{'a': 1}}\n{ATTACH1}"),
        Case("is_error text", "unchanged", res(text("boom"), is_error = True), [], "Error: boom"),
        Case("is_error empty", "unchanged", res(is_error = True), [], "Error: tool returned no content"),
        Case("is_error with image", "unchanged", res(text("boom"), image(), is_error = True), ["image/png"],
             f"Error: boom\n{ATTACH1}"),
    ]

    # -- 11. ordering and mixing -----------------------------------------------------
    C += [
        Case("image then embedded", "mixed", res(image(), blob(mime = "image/webp")),
             ["image/png", "image/webp"], ATTACH2),
        Case("embedded then image", "mixed", res(blob(mime = "image/webp"), image()),
             ["image/webp", "image/png"], ATTACH2),
        Case("text, link, image, embedded", "mixed",
             res(text("t"), rlink(), image(), blob(mime = "image/gif")),
             ["image/png", "image/gif"],
             f"t\n[resource: gen.png <file:///out/gen.png>]\n{ATTACH2}"),
        Case("embedded text + embedded image", "mixed", res(etext("note"), blob()),
             ["image/png"], f"note\n{ATTACH1}"),
    ]

    # -- 12. payload budget ----------------------------------------------------------
    huge = "A" * 12_000_001
    exact = "A" * 12_000_000
    C += [
        Case("oversized embedded image", "budget", res(blob(data = huge)), [], OMIT1),
        Case("exact budget embedded image", "budget", res(blob(data = exact)), ["image/png"], ATTACH1),
        Case("oversized then small", "budget", res(blob(data = huge), blob(mime = "image/gif")),
             ["image/gif"], f"[1 image attached; displayed to the user; 1 image omitted (too large)]"),
        Case("two images share budget", "budget", res(blob(data = exact), blob(mime = "image/gif")),
             ["image/png"], "[1 image attached; displayed to the user; 1 image omitted (too large)]"),
        Case("oversized with text", "budget", res(text("hi"), blob(data = huge)), [], f"hi\n{OMIT1}"),
    ]

    # -- 13. empty and odd payloads --------------------------------------------------
    C += [
        Case("empty blob", "odd-payload", res(blob(data = "")), [], ""),
        Case("empty data direct image", "odd-payload", res(image(data = "")), [], ""),
        Case("blob with newlines", "odd-payload", res(blob(data = "AAAA\nBBBB")), ["image/png"], ATTACH1),
        Case("blob not base64", "odd-payload", res(blob(data = "not base64 !!")), ["image/png"], ATTACH1,
             why = "flattening does not validate base64; recorded, not asserted as correct"),
    ]

    # -- 14. duck-typed / foreign objects must not crash -----------------------------
    class Bare:
        pass

    class OnlyResource:
        def __init__(self):
            self.resource = Bare()

    class ResourceWithBlobNoMime:
        def __init__(self):
            self.resource = type("R", (), {"blob": PNG_B64})()

    C += [
        Case("bare object block", "duck-typing", res(Bare()), [], ""),
        Case("block with empty resource", "duck-typing", res(OnlyResource()), [], ""),
        Case("resource blob, no mime, no uri", "duck-typing", res(ResourceWithBlobNoMime()), [], ""),
        Case("content is None", "duck-typing", FastResult(content = None, structured_content = None, meta = None, is_error = False), [], ""),
    ]

    return C


# --- running ------------------------------------------------------------------

def run(mcp_client, cases, verbose = False):
    sentinel = mcp_client.MCP_IMAGES_SENTINEL
    failures = []
    rows = []
    for case in cases:
        try:
            flat = mcp_client._flatten_result(case.result)
            err = None
        except Exception as exc:  # noqa: BLE001
            flat, err = "", f"{type(exc).__name__}: {exc}"

        if err is not None:
            rows.append((case, "RAISED", err, err))
            failures.append((case, f"raised {err}"))
            continue

        if sentinel in flat:
            body, _, payload = flat.rpartition("\n" + sentinel)
            try:
                images = json.loads(payload)
            except Exception as exc:  # noqa: BLE001
                failures.append((case, f"envelope is not valid json: {exc}"))
                rows.append((case, "BADJSON", payload[:60], flat[:60]))
                continue
        else:
            body, images = flat, []

        mimes = [img.get("mimeType") for img in images]
        problems = []

        # global invariants, checked on every case
        for img in images:
            if not isinstance(img.get("data"), str) or not isinstance(img.get("mimeType"), str):
                problems.append(f"envelope entry is not two strings: {img!r}")
            elif not CANONICAL_MIME.match(img["mimeType"]):
                problems.append(f"non-canonical mime reaches the data url: {img['mimeType']!r}")

        if case.images is not None and mimes != case.images:
            problems.append(f"images {mimes} != expected {case.images}")
        if case.body is not None and body != case.body:
            problems.append(f"body {body!r} != expected {case.body!r}")

        rows.append((case, "ok" if not problems else "FAIL", mimes, body))
        if problems:
            failures.append((case, "; ".join(problems)))

    by_group = {}
    for case, status, mimes, body in rows:
        g = by_group.setdefault(case.group, [0, 0])
        g[0] += 1
        if status == "ok":
            g[1] += 1

    print(f"{'group':<20} {'pass':>6} / {'total':<6}")
    for group, (total, passed) in by_group.items():
        mark = "" if passed == total else "   <-- failures"
        print(f"{group:<20} {passed:>6} / {total:<6}{mark}")
    print(f"{'TOTAL':<20} {len(rows) - len(failures):>6} / {len(rows):<6}")

    if verbose:
        for case, status, mimes, body in rows:
            print(f"  [{status:<4}] {case.id:<44} images={mimes} body={body[:70]!r}")

    if failures:
        print(f"\n{len(failures)} failing case(s):")
        for case, why in failures:
            print(f"  - [{case.group}] {case.id}: {why}")
            if case.why:
                print(f"      rule: {case.why}")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkout")
    ap.add_argument("--verbose", action = "store_true")
    ap.add_argument("--only", default = None)
    args = ap.parse_args()

    mcp_client = load_backend(args.checkout)

    import mcp.types as mt
    from fastmcp.client.client import CallToolResult as FastResult
    from fastmcp.utilities.types import File, Image
    from pydantic import AnyUrl
    import importlib.metadata as md

    tmp = Path(tempfile.mkdtemp(prefix = "pr9636_"))
    cases = build_cases(mt, FastResult, File, Image, AnyUrl, tmp)
    if args.only:
        cases = [c for c in cases if args.only in c.id or args.only == c.group]

    print(f"checkout : {Path(args.checkout).resolve()}")
    print(f"python   : {sys.version.split()[0]} on {sys.platform} ({os.name})")
    print(f"mcp      : {md.version('mcp')}   fastmcp: {md.version('fastmcp')}   pydantic: {md.version('pydantic')}")
    print(f"cases    : {len(cases)}\n")
    return run(mcp_client, cases, verbose = args.verbose)


if __name__ == "__main__":
    sys.exit(main())
