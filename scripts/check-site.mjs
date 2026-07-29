import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const projectRoot = resolve(import.meta.dirname, "..");
const sourceHtmlBuffer = await readFile(resolve(projectRoot, "index.html"));
const clientHtmlBuffer = await readFile(resolve(projectRoot, "dist", "client", "index.html"));
const sourceOgBuffer = await readFile(resolve(projectRoot, "public", "og.png"));
const clientOgBuffer = await readFile(resolve(projectRoot, "dist", "client", "og.png"));
const workerPath = resolve(projectRoot, "dist", "server", "index.js");
const workerSource = await readFile(workerPath, "utf8");
const html = sourceHtmlBuffer.toString("utf8");
const approvedOgSha256 = "6110243b5fa850f627c1c0ea865f00198e4e2dd62b921ce3561238332b591355";
const sourceOgSha256 = createHash("sha256").update(sourceOgBuffer).digest("hex");

if (!sourceHtmlBuffer.equals(clientHtmlBuffer)) {
  throw new Error("Built client HTML does not exactly match index.html");
}
if (!sourceOgBuffer.equals(clientOgBuffer)) {
  throw new Error("Built social preview does not exactly match public/og.png");
}
if (sourceOgSha256 !== approvedOgSha256) {
  throw new Error(
    "Social preview differs from the reviewed evergreen asset; inspect its visible " +
    "copy and update the approved digest deliberately."
  );
}

const requiredPatterns = [
  ["HTML doctype", /^<!doctype html>/i],
  ["page title", /<title>EvalSim — Closed-loop simulator evaluation<\/title>/],
  ["main landmark", /<main id="main">/],
  ["skip link", /class="skip-link" href="#main"/],
  ["policy lab", /id="policy-lab"/],
  ["evidence section", /id="evidence"/],
  ["limitations section", /id="limits"/],
  ["upstream terms section", /id="terms"/],
  ["roadmap section", /id="roadmap"/],
  ["reduced motion", /prefers-reduced-motion/],
  ["canvas text equivalent", /<table class="comparison">/],
  [
    "Waymo Open Dataset attribution",
    /This software was made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset License Agreement for Non-Commercial Use/
  ],
  [
    "Waymax prescribed notice",
    /This software was made using the Waymax Licensed Materials, provided by Waymo LLC under the Waymax License Agreement for Non-Commercial Use/
  ],
  ["Waymax prescribed spelling", /use of the Waymx Licensed Materials/],
  ["Waymax full citation", /Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research/],
  ["pinned Waymax license", /a64dfec9be8576b60d9cecc94f406d9812d4a7d0\/LICENSE/],
  ["non-commercial scope", /personal, non-commercial interview-preparation purpose/],
  ["current full-suite evidence", /757 passed/],
  ["current core-only evidence", /676 passed and 23 expected optional-runtime skips/],
  ["accepted M4 cohort evidence", /2,916 → 128/],
  ["M4 conditional-sample caveat", /conditional and nonrepresentative by design/],
  ["M4 accepted status", /M0–M4 accepted/],
  ["M4 cohort classification", /2,916 raw records: 1,527 eligible and 1,389 rejected/],
  ["locked M4 matrix evidence", /421 passed/],
  ["shared-decode limitation", /same pinned Waymax WOMD decoder/],
  ["M5 data-free evidence", /Thirteen metrics, eight source-only slices/],
  ["M5 metric boundary", /No M5 real-data scorecard or policy-quality result/],
  ["narrow benchmark scope", /batch-2 exact-log JAX kernel/],
  ["process-RSS caveat", /process high-water memory, not JAX device memory/],
  ["M5 implementation status", /Implemented data-free · M5/],
  ["M5 pending real-data gate", /bound WOMD\/Waymax metric run/],
  ["M5 synthetic command", /evalsim-m5-synthetic/],
  ["shell-safe M4 output example", /outputs\/m4\/manual-acceptance-01/],
  ["worker fetch handler", /async fetch\(request\)/]
];

for (const [name, pattern] of requiredPatterns) {
  const source = name === "worker fetch handler" ? workerSource : html;
  if (!pattern.test(source)) throw new Error(`Missing ${name}`);
}

const ids = [...html.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]);
const duplicateIds = ids.filter((id, index) => ids.indexOf(id) !== index);
if (duplicateIds.length) throw new Error(`Duplicate IDs: ${[...new Set(duplicateIds)].join(", ")}`);

const idSet = new Set(ids);
const localAnchors = [...html.matchAll(/href="#([^"]+)"/g)].map((match) => match[1]);
const brokenAnchors = localAnchors.filter((anchor) => !idSet.has(anchor));
if (brokenAnchors.length) throw new Error(`Broken local anchors: ${[...new Set(brokenAnchors)].join(", ")}`);

if (/localStorage|PROJECT_STATE|Interview Project/.test(html)) {
  throw new Error("The public presentation still contains stale tracker language or state");
}

const staleEvidencePatterns = [
  /M0–M2 complete/,
  /134 tests/,
  /M0–M3 complete/,
  /M0 → M3 implemented and tested/,
  /EvalSim · M0–M3/,
  /170 tests/,
  /170 passed/,
  /170 \/ 170/,
  /152 passed/,
  /Next · M4/,
  /M0–M4 complete/,
  /493 passed/,
  /418 passed/,
  /No M5 scorecards or metric parity/,
  /Next · M5/,
  /M5 is next/,
  /&lt;new-run-name&gt;/
];
for (const pattern of staleEvidencePatterns) {
  if (pattern.test(html)) throw new Error(`Stale presentation evidence matched ${pattern}`);
}

const imported = await import(`${pathToFileURL(workerPath).href}?check=${Date.now()}`);
const worker = imported.default;
if (!worker || typeof worker.fetch !== "function") {
  throw new Error("Built worker does not export a fetch handler");
}

const pageResponse = await worker.fetch(new Request("https://evalsim.test/"));
if (pageResponse.status !== 200 || !pageResponse.headers.get("content-type")?.startsWith("text/html")) {
  throw new Error("GET / did not return the HTML page");
}
if (await pageResponse.text() !== html) throw new Error("Worker page response is stale");
if (!pageResponse.headers.get("content-security-policy")?.includes("object-src 'none'")) {
  throw new Error("Worker response is missing the hardened content security policy");
}

const headResponse = await worker.fetch(new Request("https://evalsim.test/", { method: "HEAD" }));
if (headResponse.status !== 200 || (await headResponse.arrayBuffer()).byteLength !== 0) {
  throw new Error("HEAD / did not return an empty successful response");
}

const ogResponse = await worker.fetch(new Request("https://evalsim.test/og.png"));
const servedOgBuffer = Buffer.from(await ogResponse.arrayBuffer());
if (
  ogResponse.status !== 200 ||
  !ogResponse.headers.get("content-type")?.startsWith("image/png") ||
  !sourceOgBuffer.equals(servedOgBuffer)
) {
  throw new Error("GET /og.png did not return the current social preview");
}

const cases = [
  ["health check", new Request("https://evalsim.test/healthz"), 200],
  ["missing route", new Request("https://evalsim.test/missing"), 404],
  ["disallowed method", new Request("https://evalsim.test/", { method: "POST" }), 405]
];
for (const [name, request, expectedStatus] of cases) {
  const response = await worker.fetch(request);
  if (response.status !== expectedStatus) {
    throw new Error(`${name} returned ${response.status}; expected ${expectedStatus}`);
  }
}

console.log(
  `Site checks passed: ${ids.length} unique IDs, ${localAnchors.length} local links, ` +
  "fresh client assets, importable worker, and request matrix"
);
