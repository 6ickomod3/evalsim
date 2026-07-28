import { readFile } from "node:fs/promises";
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

if (!sourceHtmlBuffer.equals(clientHtmlBuffer)) {
  throw new Error("Built client HTML does not exactly match index.html");
}
if (!sourceOgBuffer.equals(clientOgBuffer)) {
  throw new Error("Built social preview does not exactly match public/og.png");
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
  ["current test evidence", /170 \/ 170/],
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
