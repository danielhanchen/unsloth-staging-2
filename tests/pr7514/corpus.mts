// Message corpus for the PR #7514 oracle matrix.
// Each case is the list of *visible assistant text parts* of one finished message
// (the component joins them with PART_SEPARATOR then re-splits, so >1 part
// exercises the separator).

export type Case = {
  id: string;
  parts: string[];
  note: string;
};

const FULLDOC = `<!doctype html>
<html><head><title>t</title></head><body><h1>hi</h1></body></html>`;

const FULLDOC_HTMLTAG = `<html lang="en">
<body><p>no doctype, still a full document</p></body>
</html>`;

const FRAGMENT = `<div class="card"><p>just a fragment</p></div>`;

export const CORPUS: Case[] = [
  // ---- full documents, the path the in-place collapse owns -------------------
  {
    id: "fulldoc-plain",
    parts: ["Here you go:\n\n```html\n" + FULLDOC + "\n```\n"],
    note: "plain 3-backtick unindented full document (the canonical case)",
  },
  {
    id: "fulldoc-htmltag",
    parts: ["```html\n" + FULLDOC_HTMLTAG + "\n```"],
    note: "<html> opener, no doctype -> still isFullDocument",
  },
  {
    id: "fulldoc-html4-public",
    parts: [
      '```html\n<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">\n<html><body>x</body></html>\n```',
    ],
    note: "HTML4 public doctype; \\b after html means this still matches",
  },
  {
    id: "fulldoc-uppercase-doctype",
    parts: ["```html\n<!DOCTYPE html>\n<html><body>x</body></html>\n```"],
    note: "uppercase DOCTYPE, regex is case-insensitive",
  },
  {
    id: "fulldoc-leading-blankline",
    parts: ["```html\n\n\n" + FULLDOC + "\n```"],
    note: "leading blank lines; trimStart should still see the doctype",
  },
  {
    id: "fulldoc-leading-comment",
    parts: ["```html\n<!-- a comment first -->\n" + FULLDOC + "\n```"],
    note: "leading HTML comment defeats isFullHtmlDocument -> treated as a fragment",
  },
  {
    id: "fulldoc-crlf",
    parts: ["```html\r\n" + FULLDOC.replace(/\n/g, "\r\n") + "\r\n```"],
    note: "CRLF line endings (Windows-authored / Windows-proxied model output)",
  },
  {
    id: "fulldoc-trailing-space-on-close",
    parts: ["```html\n" + FULLDOC + "\n```   "],
    note: "close fence with trailing spaces; closeRe allows \\s*",
  },

  // ---- non-plain fences: 4+ backticks and indentation ------------------------
  {
    id: "fulldoc-4backtick",
    parts: ["````html\n" + FULLDOC + "\n````"],
    note: "4-backtick fence: NOT plain, so the in-place collapse never sees it",
  },
  {
    id: "fulldoc-5backtick",
    parts: ["`````html\n" + FULLDOC + "\n`````"],
    note: "5-backtick fence",
  },
  {
    id: "fulldoc-indent1",
    parts: [" ```html\n " + FULLDOC.replace(/\n/g, "\n ") + "\n ```"],
    note: "1-space indented fence: NOT plain",
  },
  {
    id: "fulldoc-indent3",
    parts: ["   ```html\n   " + FULLDOC.replace(/\n/g, "\n   ") + "\n   ```"],
    note: "3-space indented fence (max CommonMark indent): NOT plain",
  },
  {
    id: "fulldoc-indent4",
    parts: ["    ```html\n    " + FULLDOC + "\n    ```"],
    note: "4-space indent: not a fence at all (indented code block)",
  },
  {
    id: "fulldoc-4backtick-closed-by-more",
    parts: ["````html\n" + FULLDOC + "\n``````"],
    note: "closed by MORE backticks than opened; CommonMark-legal",
  },

  // ---- fragments: the in-place collapse never handles these -------------------
  {
    id: "fragment-div",
    parts: ["```html\n" + FRAGMENT + "\n```"],
    note: "bare <div> fragment: card is/was its ONLY preview path",
  },
  {
    id: "fragment-table",
    parts: ["```html\n<table><tr><td>1</td></tr></table>\n```"],
    note: "table fragment",
  },
  {
    id: "fragment-style",
    parts: ["```html\n<style>body{color:red}</style>\n```"],
    note: "style-only fragment",
  },
  {
    id: "fragment-script-with-backticks",
    parts: [
      "```html\n<script>const s = `a ``` b`; console.log(s);</script>\n```",
    ],
    note: "backticks inside a <script> template literal must not close the fence",
  },
  {
    id: "fragment-body-only",
    parts: ["```html\n<body><p>x</p></body>\n```"],
    note: "<body> without <html>: fragment",
  },

  // ---- SVG is deliberately routed away from artifacts ------------------------
  {
    id: "svg-in-html-fence",
    parts: ['```html\n<svg viewBox="0 0 1 1"><rect/></svg>\n```'],
    note: "SVG inside an html fence -> dropped by isSvgFence, never a card",
  },
  {
    id: "svg-lang",
    parts: ["```svg\n<svg><rect/></svg>\n```"],
    note: "svg lang -> not an html fence",
  },
  {
    id: "xml-svg",
    parts: ['```html\n<?xml version="1.0"?>\n<svg><rect/></svg>\n```'],
    note: "xml prolog + svg -> isSvgFence true",
  },

  // ---- language-tag variants -------------------------------------------------
  {
    id: "lang-uppercase-HTML",
    parts: ["```HTML\n" + FULLDOC + "\n```"],
    note: "uppercase lang; extractHtmlFences lowercases, CODE_FENCE_RE does not",
  },
  {
    id: "lang-htm",
    parts: ["```htm\n" + FULLDOC + "\n```"],
    note: "htm is NOT recognised as html",
  },
  {
    id: "lang-xhtml",
    parts: ["```xhtml\n" + FULLDOC + "\n```"],
    note: "xhtml is NOT recognised",
  },
  {
    id: "lang-none",
    parts: ["```\n" + FULLDOC + "\n```"],
    note: "untagged fence -> no lang -> no card",
  },
  {
    id: "lang-html-with-meta",
    parts: ["```html title=index.html\n" + FULLDOC + "\n```"],
    note: "info string with extra metadata; first token is html",
  },
  {
    id: "lang-html-trailing-space",
    parts: ["```html \n" + FULLDOC + "\n```"],
    note: "trailing space after the lang tag",
  },

  // ---- degenerate / adversarial ---------------------------------------------
  {
    id: "unterminated-fence",
    parts: ["```html\n" + FULLDOC],
    note: "unterminated fence (truncated generation) -> extractHtmlFences breaks, 0 cards",
  },
  {
    id: "empty-fence",
    parts: ["```html\n```"],
    note: "empty html fence",
  },
  {
    id: "whitespace-only-fence",
    parts: ["```html\n   \n\n```"],
    note: "whitespace-only body",
  },
  {
    id: "no-fence-at-all",
    parts: ["Just a plain sentence with no code in it."],
    note: "control: no fences",
  },
  {
    id: "inline-backticks-only",
    parts: ["Use `<div>` inline, and ``code`` too."],
    note: "inline code spans must not be treated as fences",
  },
  {
    id: "prose-then-fence-then-prose",
    parts: ["Intro.\n\n```html\n" + FRAGMENT + "\n```\n\nOutro text."],
    note: "fence surrounded by prose",
  },

  // ---- multiple fences -------------------------------------------------------
  {
    id: "two-fulldocs",
    parts: ["```html\n" + FULLDOC + "\n```\n\n```html\n" + FULLDOC + "\n```"],
    note: "two full documents -> two cards when enabled",
  },
  {
    id: "fulldoc-plus-fragment",
    parts: ["```html\n" + FULLDOC + "\n```\n\n```html\n" + FRAGMENT + "\n```"],
    note: "mixed: one collapsible full doc + one fragment",
  },
  {
    id: "html-plus-python",
    parts: [
      "```python\nprint(1)\n```\n\n```html\n" + FRAGMENT + "\n```",
    ],
    note: "non-html fence must be ignored entirely",
  },
  {
    id: "three-fences-mixed",
    parts: [
      "```html\n" +
        FULLDOC +
        "\n```\n\n```js\nx\n```\n\n````html\n" +
        FRAGMENT +
        "\n````",
    ],
    note: "plain full doc + js + non-plain fragment",
  },

  // ---- multi-part messages: the whole point of PART_SEPARATOR ----------------
  {
    id: "split-fence-across-parts",
    parts: ["```html\n" + FRAGMENT, "\n```\nrest of the answer"],
    note: "CRITICAL: fence opened in part 1, closed in part 2 -> must NOT stitch into a card",
  },
  {
    id: "one-fence-per-part",
    parts: [
      "```html\n" + FULLDOC + "\n```",
      "```html\n" + FRAGMENT + "\n```",
    ],
    note: "two parts each with a complete fence -> both found",
  },
  {
    id: "fence-in-second-part-only",
    parts: ["Reasoning-adjacent prose.", "```html\n" + FRAGMENT + "\n```"],
    note: "fence only in the trailing text part",
  },
  {
    id: "three-parts-middle-fence",
    parts: ["a", "```html\n" + FULLDOC + "\n```", "b"],
    note: "three text parts, fence in the middle",
  },
];
