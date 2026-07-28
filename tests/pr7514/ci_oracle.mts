// Cross-OS oracle for unslothai/unsloth PR #7514.
//
// Self-contained: html-fences.ts is byte-identical before and after the PR, so
// this imports the repo's own copy once and applies BOTH the pre-PR and post-PR
// card predicates to it. No node_modules required -- runs on node's native TS
// type stripping, which is exactly what we want to confirm on Linux/macOS/Windows.
//
// Run: node --experimental-strip-types tests/pr7514/ci_oracle.mts

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { CORPUS } from "./corpus.mts";
import {
  extractHtmlFences,
  getCodeFence,
  isFullHtmlDocument,
  isHtmlFence,
} from "../../studio/frontend/src/features/chat/artifacts/html-fences.ts";

const SEP = "\u0000";

type Flags = {
  artifactsEnabled: boolean;
  collapseHtmlArtifacts: boolean;
  loadedIsDiffusion: boolean;
  hasRenderHtmlTool: boolean;
  isRunning: boolean;
};

const scan = (parts: string[]) =>
  parts
    .join(SEP)
    .split(SEP)
    .flatMap((p) => extractHtmlFences(p));

/** pre-PR: toggles only chose what to SKIP */
function oldCards(parts: string[], f: Flags) {
  const collapses =
    (f.artifactsEnabled || f.collapseHtmlArtifacts) && !f.loadedIsDiffusion;
  if (f.isRunning || f.hasRenderHtmlTool) return [];
  return scan(parts).filter(
    (x) => !(x.isFullDocument && x.isPlainFence && collapses),
  );
}

/** post-PR: cards require at least one toggle */
function newCards(parts: string[], f: Flags) {
  const enabled = f.artifactsEnabled || f.collapseHtmlArtifacts;
  const collapses = enabled && !f.loadedIsDiffusion;
  if (f.isRunning || f.hasRenderHtmlTool || !enabled) return [];
  return scan(parts).filter(
    (x) => !(x.isFullDocument && x.isPlainFence && collapses),
  );
}

const COMBOS: Flags[] = [];
for (const a of [false, true])
  for (const c of [false, true])
    for (const d of [false, true])
      for (const t of [false, true])
        for (const r of [false, true])
          COMBOS.push({
            artifactsEnabled: a,
            collapseHtmlArtifacts: c,
            loadedIsDiffusion: d,
            hasRenderHtmlTool: t,
            isRunning: r,
          });

const failures: string[] = [];
let rows = 0;
let changed = 0;
const changedCombos = new Set<string>();
const fk = (f: Flags) =>
  `${f.artifactsEnabled ? "A" : "-"}${f.collapseHtmlArtifacts ? "C" : "-"}${f.loadedIsDiffusion ? "D" : "-"}${f.hasRenderHtmlTool ? "T" : "-"}${f.isRunning ? "R" : "-"}`;

for (const c of CORPUS) {
  for (const f of COMBOS) {
    rows++;
    const o = oldCards(c.parts, f).length;
    const n = newCards(c.parts, f).length;
    if (o !== n) {
      changed++;
      changedCombos.add(fk(f));
      // Every difference must be a case where BOTH toggles are off. If a change
      // ever shows up with a toggle ON, the PR has widened its blast radius.
      if (f.artifactsEnabled || f.collapseHtmlArtifacts) {
        failures.push(
          `case=${c.id} flags=${fk(f)}: cards changed ${o}->${n} while a toggle is ON`,
        );
      }
    }
    // With a toggle on, post-PR must equal pre-PR exactly.
    if ((f.artifactsEnabled || f.collapseHtmlArtifacts) && o !== n) {
      failures.push(`regression: ${c.id} ${fk(f)}`);
    }
    // The post-PR set must always be a subset of the pre-PR set.
    const oldSrc = new Set(oldCards(c.parts, f).map((x) => x.source));
    for (const x of newCards(c.parts, f)) {
      if (!oldSrc.has(x.source)) {
        failures.push(`new card not present pre-PR: ${c.id} ${fk(f)}`);
      }
    }
  }
}

console.log("=".repeat(78));
console.log(`PR #7514 cross-OS oracle`);
console.log(
  `platform=${os.platform()} arch=${os.arch()} node=${process.version} eol=${JSON.stringify(os.EOL)}`,
);
console.log("=".repeat(78));
console.log(`cases=${CORPUS.length} combos=${COMBOS.length} rows=${rows}`);
console.log(`rows changed by the PR: ${changed}`);
console.log(
  `flag combos that change:  ${[...changedCombos].sort().join(" ") || "(none)"}`,
);

// ---- separator semantics ---------------------------------------------------
const sepOk =
  SEP.length === 1 &&
  SEP.charCodeAt(0) === 0 &&
  ["a", "b"].join(SEP).split(SEP).length === 2;
console.log(`separator is a single NUL and round-trips: ${sepOk}`);
if (!sepOk) failures.push("separator semantics broken on this platform");

// A fence must never be stitched across two text parts.
const stitch = CORPUS.find((c) => c.id === "split-fence-across-parts")!;
const on: Flags = {
  artifactsEnabled: true,
  collapseHtmlArtifacts: true,
  loadedIsDiffusion: false,
  hasRenderHtmlTool: false,
  isRunning: false,
};
if (newCards(stitch.parts, on).length !== 0)
  failures.push("fence stitched across parts");
if (extractHtmlFences(stitch.parts.join("")).length !== 1)
  failures.push("stitch control case did not behave as expected");
console.log(`fence never stitches across message parts: true`);

// ---- on-disk file hygiene (the genuinely OS-sensitive part) -----------------
const target = path.join(
  "studio",
  "frontend",
  "src",
  "components",
  "assistant-ui",
  "message-html-artifacts.tsx",
);
const buf = fs.readFileSync(target);
const nuls = buf.filter((b) => b === 0).length;
const crs = buf.filter((b) => b === 13).length;
console.log(
  `checked-out file: ${buf.length} bytes, NUL=${nuls}, CR=${crs} (expect NUL=0, CR=0 via .gitattributes eol=lf)`,
);
if (nuls !== 0) failures.push(`file still contains ${nuls} NUL byte(s)`);
if (crs !== 0)
  failures.push(
    `file checked out with ${crs} CR byte(s); .gitattributes eol=lf should prevent this`,
  );

// CRLF-in-message robustness, which is what a Windows-authored answer looks like.
const crlfCase = CORPUS.find((c) => c.id === "fulldoc-crlf")!;
const crlfCards = newCards(crlfCase.parts, on).length;
const crlfOff = newCards(crlfCase.parts, {
  ...on,
  artifactsEnabled: false,
  collapseHtmlArtifacts: false,
}).length;
console.log(`CRLF message body: cards with toggles on=${crlfCards} off=${crlfOff}`);
if (crlfCards !== 0 || crlfOff !== 0)
  failures.push("CRLF full-document handling diverged on this platform");

console.log("=".repeat(78));
if (failures.length) {
  console.log(`FAIL: ${failures.length} problem(s)`);
  for (const f of failures.slice(0, 30)) console.log("  - " + f);
  process.exit(1);
}
console.log("PASS: all assertions hold on this platform");
