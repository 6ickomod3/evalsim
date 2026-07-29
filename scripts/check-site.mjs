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
  ["current full-suite evidence", /876 passed/],
  ["current full-suite local skip", /1 expected local-data skip/],
  ["current core-only evidence", /790 passed with 28 expected optional\/local skips/],
  ["accepted M4 cohort evidence", /2,916 → 128/],
  ["M4 conditional-sample caveat", /conditional and nonrepresentative by design/],
  ["M5 accepted status", /M0–M5 accepted/],
  ["M4 cohort classification", /2,916 raw records: 1,527 eligible and 1,389 rejected/],
  ["shared-decode limitation", /same pinned Waymax WOMD decoder/],
  ["M5 official ten-shard evidence", /bound ten-shard run completed on all 128 accepted scenarios/],
  [
    "M5 official result domains",
    /6,656 metric rows, 1,024 slice memberships, 312 scorecards, and 144 native-parity summaries/
  ],
  [
    "M5 primary completeness",
    /All 12 primary cells retained paired n = 128 with zero exclusions or asymmetric missingness/
  ],
  ["M5 bounded parity axes", /16 scenes × 20 post-current transitions × three policies/],
  ["M5 parity component evidence", /38,754 overlap components and 37,770 fixed-step kinematic components had zero binary mismatches/],
  ["M5 log-divergence parity nuance", /24 of 48 rows had nonzero error within the pre-registered float32 tolerance/],
  ["M5 complete primary matrix", /Complete 12-cell M5 primary all-slice matrix/],
  ["M5 metric-contract field", /Metric contract/],
  ["M5 pairing field", /Pairing \+ missingness/],
  ["M5 exact-effect field", /Exact raw mean \/ N/],
  ["M5 adjusted primary level", /Adjusted 0\.9958333333333333 band/],
  ["M5 sparsity field", /Nonzero \/ status \/ suppression/],
  ["M5 separate raw sign", /Raw sign/],
  ["M5 separate oriented sign", /Oriented advantage sign/],
  ["M5 held overlap result", /bands cross zero/],
  ["M5 held kinematic result", /event-sparse; adjusted crosses zero/],
  ["M5 uncertainty boundary", /not confidence intervals, hypothesis tests, or population claims/],
  ["M5 no-winner boundary", /Every contrast has metrics pointing in both directions/],
  ["narrow benchmark scope", /batch-2 exact-log JAX kernel/],
  ["process-RSS caveat", /process high-water memory, not JAX device memory/],
  ["M5 implementation status", /Implemented \+ accepted · M5/],
  ["M6 next status", /Next falsifiable question \/ M6 not started/],
  ["M5 synthetic command", /evalsim-m5-synthetic/],
  ["M5 official command", /evalsim-m5-official/],
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

const resultTable = html.match(
  /<table class="comparison result-table">([\s\S]*?)<\/table>/
)?.[1];
if (!resultTable) throw new Error("Missing complete M5 primary result table");
const resultBody = resultTable.match(/<tbody>([\s\S]*?)<\/tbody>/)?.[1];
const resultRows = resultBody?.match(/<tr>/g)?.length ?? 0;
if (resultRows !== 12) {
  throw new Error(`M5 primary result table has ${resultRows} rows; expected 12`);
}
const directionalGates = resultBody?.match(/class="gate-pass"/g)?.length ?? 0;
const heldGates = resultBody?.match(/class="gate-hold"/g)?.length ?? 0;
if (directionalGates !== 10 || heldGates !== 2) {
  throw new Error(
    `M5 primary evidence gates are ${directionalGates} directional and ${heldGates} held; expected 10 and 2`
  );
}

const resultRowBodies = [...resultBody.matchAll(/<tr>([\s\S]*?)<\/tr>/g)].map(
  (match) => match[1]
);
const sharedValidity =
  "paired 128<br>valid A/B 128/128<br>excluded/asymmetric 0/0<br>" +
  "both-missing 0<br>reasons A/B {}/{}";
const expectedPrimaryRows = [
  {
    contract: "position_error_m</code><br><span>v1.0.0 · m · lower",
    contrast: "<code>constant_velocity</code><br>− <code>log_replay</code>",
    raw: "2.997305058703697<br>N = 128",
    pointwise: "[2.6279802249751203,<br>3.3901418187934973]",
    adjusted: "[2.4703977791024374,<br>3.5883347567281167]",
    sparse: "nonzero 128<br><code>descriptive</code><br>suppressed no",
    rawSign: '<td class="numeric">positive</td>',
    oriented: "-2.997305058703697<br>negative · A worse",
    gate: 'class="gate-pass">directional yes'
  },
  {
    contract: "position_error_m</code><br><span>v1.0.0 · m · lower",
    contrast: "<code>idm</code><br>− <code>log_replay</code>",
    raw: "7.7500647664453<br>N = 128",
    pointwise: "[7.254792072654283,<br>8.238701486809102]",
    adjusted: "[7.039911175083419,<br>8.46896905650854]",
    sparse: "nonzero 128<br><code>descriptive</code><br>suppressed no",
    rawSign: '<td class="numeric">positive</td>',
    oriented: "-7.7500647664453<br>negative · A worse",
    gate: 'class="gate-pass">directional yes'
  },
  {
    contract: "position_error_m</code><br><span>v1.0.0 · m · lower",
    contrast: "<code>idm</code><br>− <code>constant_velocity</code>",
    raw: "4.752759707741603<br>N = 128",
    pointwise: "[4.202246972287565,<br>5.3075581383441]",
    adjusted: "[3.95256553101306,<br>5.56545970470591]",
    sparse: "nonzero 128<br><code>descriptive</code><br>suppressed no",
    rawSign: '<td class="numeric">positive</td>',
    oriented: "-4.752759707741603<br>negative · A worse",
    gate: 'class="gate-pass">directional yes'
  },
  {
    contract: "speed_error_mps</code><br><span>v1.0.0 · m/s · lower",
    contrast: "<code>constant_velocity</code><br>− <code>log_replay</code>",
    raw: "1.0965694309637917<br>N = 128",
    pointwise: "[0.9619761038524979,<br>1.2391368628083081]",
    adjusted: "[0.9032779039639633,<br>1.3053634771162963]",
    sparse: "nonzero 128<br><code>descriptive</code><br>suppressed no",
    rawSign: '<td class="numeric">positive</td>',
    oriented: "-1.0965694309637917<br>negative · A worse",
    gate: 'class="gate-pass">directional yes'
  },
  {
    contract: "speed_error_mps</code><br><span>v1.0.0 · m/s · lower",
    contrast: "<code>idm</code><br>− <code>log_replay</code>",
    raw: "2.9496225910279303<br>N = 128",
    pointwise: "[2.7967982543093335,<br>3.104855524781929]",
    adjusted: "[2.7243675139407824,<br>3.1765169409116614]",
    sparse: "nonzero 128<br><code>descriptive</code><br>suppressed no",
    rawSign: '<td class="numeric">positive</td>',
    oriented: "-2.9496225910279303<br>negative · A worse",
    gate: 'class="gate-pass">directional yes'
  },
  {
    contract: "speed_error_mps</code><br><span>v1.0.0 · m/s · lower",
    contrast: "<code>idm</code><br>− <code>constant_velocity</code>",
    raw: "1.8530531600641384<br>N = 128",
    pointwise: "[1.6431704146322648,<br>2.063347933439761]",
    adjusted: "[1.5420387726424956,<br>2.1559393836500935]",
    sparse: "nonzero 128<br><code>descriptive</code><br>suppressed no",
    rawSign: '<td class="numeric">positive</td>',
    oriented: "-1.8530531600641384<br>negative · A worse",
    gate: 'class="gate-pass">directional yes'
  },
  {
    contract: "oriented_box_overlap_rate</code><br><span>v1.0.0 · fraction · lower",
    contrast: "<code>constant_velocity</code><br>− <code>log_replay</code>",
    raw: "0.04966195639263592<br>N = 128",
    pointwise: "[0.03709369889680477,<br>0.06256488985435854]",
    adjusted: "[0.031503882406754925,<br>0.06888376611885266]",
    sparse: "nonzero 112<br><code>descriptive</code><br>suppressed no",
    rawSign: '<td class="numeric">positive</td>',
    oriented: "-0.04966195639263592<br>negative · A worse",
    gate: 'class="gate-pass">directional yes'
  },
  {
    contract: "oriented_box_overlap_rate</code><br><span>v1.0.0 · fraction · lower",
    contrast: "<code>idm</code><br>− <code>log_replay</code>",
    raw: "0.04324182820631069<br>N = 128",
    pointwise: "[0.034523267487685035,<br>0.05171025074417448]",
    adjusted: "[0.03023094433374868,<br>0.05548078521175095]",
    sparse: "nonzero 117<br><code>descriptive</code><br>suppressed no",
    rawSign: '<td class="numeric">positive</td>',
    oriented: "-0.04324182820631069<br>negative · A worse",
    gate: 'class="gate-pass">directional yes'
  },
  {
    contract: "oriented_box_overlap_rate</code><br><span>v1.0.0 · fraction · lower",
    contrast: "<code>idm</code><br>− <code>constant_velocity</code>",
    raw: "-0.006420128186325231<br>N = 128",
    pointwise: "[-0.015279928885177754,<br>0.0020568185134115127]",
    adjusted: "[-0.01956296032512017,<br>0.005847794738256519]",
    sparse: "nonzero 117<br><code>descriptive</code><br>suppressed no",
    rawSign: '<td class="numeric">negative</td>',
    oriented: "0.006420128186325231<br>positive · A better",
    gate: 'class="gate-hold">directional no<br>bands cross zero'
  },
  {
    contract:
      "waymax_kinematic_infeasibility_rate</code><br><span>v1.0.1 · fraction · lower",
    contrast: "<code>constant_velocity</code><br>− <code>log_replay</code>",
    raw: "-0.04413296796842321<br>N = 128",
    pointwise: "[-0.04984095659342634,<br>-0.038919178663599646]",
    adjusted: "[-0.052764339963662525,<br>-0.036711842617360955]",
    sparse: "nonzero 128<br><code>descriptive</code><br>suppressed no",
    rawSign: '<td class="numeric">negative</td>',
    oriented: "0.04413296796842321<br>positive · A better",
    gate: 'class="gate-pass">directional yes'
  },
  {
    contract:
      "waymax_kinematic_infeasibility_rate</code><br><span>v1.0.1 · fraction · lower",
    contrast: "<code>idm</code><br>− <code>log_replay</code>",
    raw: "-0.044146410267774536<br>N = 128",
    pointwise: "[-0.04983342988405147,<br>-0.03893367324809029]",
    adjusted: "[-0.05276528621539924,<br>-0.036806857987709767]",
    sparse: "nonzero 128<br><code>descriptive</code><br>suppressed no",
    rawSign: '<td class="numeric">negative</td>',
    oriented: "0.044146410267774536<br>positive · A better",
    gate: 'class="gate-pass">directional yes'
  },
  {
    contract:
      "waymax_kinematic_infeasibility_rate</code><br><span>v1.0.1 · fraction · lower",
    contrast: "<code>idm</code><br>− <code>constant_velocity</code>",
    raw: "-1.3442299351326267e-05<br>N = 128",
    pointwise: "[-2.7997259380531403e-05,<br>-1.0808611332096852e-06]",
    adjusted: "[-3.6473133948291926e-05,<br>3.2271037770856812e-06]",
    sparse: "nonzero 7<br><code>event_sparse</code><br>suppressed no",
    rawSign: '<td class="numeric">negative</td>',
    oriented: "1.3442299351326267e-05<br>positive · A better",
    gate:
      'class="gate-hold">directional no<br>event-sparse; adjusted crosses zero'
  }
];
for (const [index, expected] of expectedPrimaryRows.entries()) {
  const row = resultRowBodies[index];
  const fields = { validity: sharedValidity, ...expected };
  const missingFields = Object.entries(fields)
    .filter(([, fragment]) => !row.includes(fragment))
    .map(([name]) => name);
  const visibleCells = row.match(/<(?:th|td)\b/g)?.length ?? 0;
  if (visibleCells !== 10 || missingFields.length) {
    throw new Error(
      `M5 primary row ${index + 1} has ${visibleCells} visible fields; ` +
      `missing or changed: ${missingFields.join(", ") || "none"}`
    );
  }
}

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
  /757 passed/,
  /No M5 scorecards or metric parity/,
  /Implemented data-free · M5/,
  /Next · M5/,
  /M5 is next/,
  /M5 official runner accepted data-free/,
  /real M5 WOMD run pending/,
  /runner has not yet produced a WOMD metric outcome/i,
  /No M5 real-data scorecard/,
  /real WOMD\/Waymax outcomes remain pending/,
  /no M5 WOMD result is represented/i,
  /Bound WOMD\/Waymax run/,
  /no result yet/,
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
