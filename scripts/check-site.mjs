import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const projectRoot = resolve(import.meta.dirname, "..");
const sourceHtmlBuffer = await readFile(resolve(projectRoot, "index.html"));
const clientHtmlBuffer = await readFile(resolve(projectRoot, "dist", "client", "index.html"));
const sourceInterviewBuffer = await readFile(resolve(projectRoot, "interview-prep.html"));
const clientInterviewBuffer = await readFile(
  resolve(projectRoot, "dist", "client", "interview-prep.html")
);
const sourceOgBuffer = await readFile(resolve(projectRoot, "public", "og.png"));
const clientOgBuffer = await readFile(resolve(projectRoot, "dist", "client", "og.png"));
const workerPath = resolve(projectRoot, "dist", "server", "index.js");
const workerSource = await readFile(workerPath, "utf8");
const html = sourceHtmlBuffer.toString("utf8");
const interviewHtml = sourceInterviewBuffer.toString("utf8");
const approvedOgSha256 = "6110243b5fa850f627c1c0ea865f00198e4e2dd62b921ce3561238332b591355";
const sourceOgSha256 = createHash("sha256").update(sourceOgBuffer).digest("hex");

if (!sourceHtmlBuffer.equals(clientHtmlBuffer)) {
  throw new Error("Built client HTML does not exactly match index.html");
}
if (!sourceInterviewBuffer.equals(clientInterviewBuffer)) {
  throw new Error("Built interview guide does not exactly match interview-prep.html");
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
  ["page title", /<title>EvalSim — Metrics-first simulator evaluation<\/title>/],
  ["metrics-first hero", /The simulator ranking depends on <em>what you measure\.<\/em>/],
  ["three-baseline four-metric identity", /Three baselines · four primary metrics · one fixed 128-scene conditional cohort/],
  ["precise hero status", /<span class="status-live">M5 accepted<\/span>/],
  ["learning-project boundary", /personal learning project and work in progress/],
  ["interview guide link", /href="\/interview"/],
  ["current full-suite evidence", /1,444 passed/],
  ["current full-suite local skip", /1 expected local-data skip/],
  ["main landmark", /<main id="main">/],
  ["skip link", /class="skip-link" href="#main"/],
  ["policy lab", /id="policy-lab"/],
  ["metrics section", /id="metrics"/],
  ["M7 stress-test section", /id="stress-test"/],
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
  ["M5 snapshot full-suite evidence", /876 passed/],
  ["M5 snapshot local skip", /1 expected local-data skip/],
  ["M5 snapshot core-only evidence", /790 passed with 28 expected optional\/local skips/],
  ["accepted M4 cohort evidence", /2,916 → 128/],
  ["M4 conditional-sample caveat", /conditional and nonrepresentative by design/],
  ["M5 accepted status", /M5 comparison accepted/],
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
  [
    "metrics-first conclusion",
    /There is no overall winner: changing the evaluator changes the ordering\./
  ],
  ["paired scenario unit", /The comparison unit is the scenario—not an agent or frame\./],
  ["source-only eligibility", /Eligibility and slice membership use source data only, before policy outcomes/],
  ["no composite score", /not a composite realism score or simulator champion/],
  ["conditional cohort headline", /fixed 128-scene conditional cohort/],
  ["narrow benchmark scope", /batch-2 exact-log JAX kernel/],
  ["process-RSS caveat", /process high-water memory, not JAX device memory/],
  ["M5 implementation status", /Implemented \+ accepted · M5/],
  ["M7 completed construct-audit status", /Completed bounded construct audit · M7/],
  ["M7 construct-audit evidence ID", /data-evidence="m7-construct-audit"/],
  [
    "M7 outcome-aware analytic scope",
    /This outcome-aware audit used exactly three hand-built analytic cases at four doses/
  ],
  [
    "M7 v1 current-frame artifact",
    /v1 freeze changed velocity at the observed current frame, erasing the future deceleration/
  ],
  [
    "M7 v2 abrupt-stop correction",
    /Freeze v2 preserves history and the current frame, then introduces a future-only abrupt stop from the next frame/
  ],
  ["M7 nonreactivity claim boundary", /not generic nonreactivity/],
  [
    "M7 validation boundary",
    /not calibrated, source-disjoint, held-out, WOMD-backed, population-level, or general metric validation/
  ],
  ["M7 speed exclusion", /speed_error_mps<\/code> is excluded from this audit/],
  ["M7 incomplete boundary", /original M7 v1 gates remain unmet/],
  ["M7 optional extension", /broader validation is optional and separately gated/],
  ["M6 pre-data status", /Implemented \(pre-data\) · M6/],
  ["extension status", /Extensions <span>separately gated<\/span>/],
  ["M6 no-result boundary", /No official WOMD result is accepted/],
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

const interviewRequiredPatterns = [
  ["HTML doctype", /^<!doctype html>/i],
  [
    "page title",
    /<title>EvalSim Interview Lab — Learn it, explain it, defend it\.<\/title>/
  ],
  [
    "page heading",
    /<h1[^>]*>\s*EvalSim Interview Lab — Learn it, explain it, defend it\.?\s*<\/h1>/
  ],
  ["main landmark", /<main id="main"/],
  ["skip link", /class="skip-link" href="#main"/],
  ["presentation back-link", /href="\/"/],
  ["domain-knowledge boundary", /No prior autonomous-driving domain knowledge required\./],
  ["learning-project boundary", /Personal learning project and work in progress/],
  ["public-study-guide boundary", /Public, sanitized study guide/],
  ["private-note exclusion", /no private interview notes/i],
  ["contact-information exclusion", /contact (?:details|information)/i],
  ["scheduling-information exclusion", /scheduling information/i],
  ["dataset-payload exclusion", /dataset payloads/i],
  ["non-commercial scope", /personal, non-commercial interview-preparation purpose/],
  ["current full-suite evidence", /1,444 passed/],
  ["current full-suite skip", /1 skipped/],
  ["M5 accepted boundary", /M5[^<]{0,80}(?:complete[^<]{0,30}accepted|accepted[^<]{0,30}complete)/i],
  ["M7 bounded boundary", /M7[^<]{0,100}bounded construct (?:audit|evidence)/i],
  ["M6 result boundary", /M6[^<]{0,100}no accepted WOMD result/i],
  ["log-replay baseline", /log replay/i],
  ["constant-velocity baseline", /constant velocity/i],
  ["IDM expansion", /Intelligent Driver Model/],
  ["ego definition", /ego/i],
  ["position metric", /position_error_m/],
  ["speed metric", /speed_error_mps/],
  ["overlap metric", /oriented_box_overlap_rate/],
  ["kinematic metric", /waymax_kinematic_infeasibility_rate/],
  ["no-winner boundary", /no overall winner/i],
  ["no-composite boundary", /(?:did not create|not|no) (?:a )?composite realism score/i],
  ["M7 held-out boundary", /not calibrated, source-disjoint, held-out, WOMD-backed/i],
  ["M7 nonreactivity boundary", /(?:not|does not|prove) generic nonreactivity/i],
  ["progress storage key", /evalsim-interview-progress-v1/],
  ["learn-mode interaction", /id="learn-mode"/],
  ["drill-mode interaction", /id="drill-mode"/],
  ["glossary-filter interaction", /id="glossary-filter"/],
  ["glossary items", /class="glossary-item"/],
  ["glossary live status", /id="glossary-status"/],
  ["30-second tab", /id="tab-30"/],
  ["2-minute tab", /id="tab-2"/],
  ["10-minute tab", /id="tab-10"/],
  ["30-second panel", /id="panel-30"/],
  ["2-minute panel", /id="panel-2"/],
  ["10-minute panel", /id="panel-10"/],
  ["clear-progress interaction", /id="clear-progress"/],
  ["new-question interaction", /id="new-question"/],
  ["reveal-answer interaction", /id="reveal-answer"/],
  ["random-question interaction", /id="random-question"/],
  ["random-answer interaction", /id="random-answer"/],
  ["copy-track interaction", /copy-track/],
  ["copy target wiring", /data-copy-target/],
  ["copy live status", /id="copy-status"/],
  ["saved progress controls", /data-progress-id/],
  ["local progress read", /localStorage\.getItem\(STORAGE_KEY\)/],
  ["local progress write", /localStorage\.setItem\(STORAGE_KEY/],
  ["clipboard implementation", /navigator\.clipboard\.writeText/],
  ["random-question implementation", /Math\.random\(\)/],
  ["live interaction feedback", /aria-live/],
  ["reduced motion", /prefers-reduced-motion/],
  [
    "Waymo Open Dataset attribution",
    /This software was made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset License Agreement for Non-Commercial Use/
  ],
  [
    "Waymax prescribed notice",
    /This software was made using the Waymax Licensed Materials, provided by Waymo LLC under the Waymax License Agreement for Non-Commercial Use/
  ],
  ["Waymax prescribed spelling", /use of the Waymx Licensed Materials/],
  [
    "Waymax full citation",
    /Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research/
  ],
  ["pinned Waymax license", /a64dfec9be8576b60d9cecc94f406d9812d4a7d0\/LICENSE/]
];
for (const [name, pattern] of interviewRequiredPatterns) {
  if (!pattern.test(interviewHtml)) throw new Error(`Interview guide is missing ${name}`);
}

const expectedInterviewSections = [
  "orientation",
  "glossary",
  "system",
  "baselines",
  "metrics",
  "experiment",
  "m5-result",
  "m7-audit",
  "talk-tracks",
  "interview-drill",
  "flashcards",
  "pitfalls",
  "last-day",
  "sources"
];
let previousSectionIndex = -1;
for (const sectionId of expectedInterviewSections) {
  const sectionPattern = new RegExp(`<section[^>]*\\sid="${sectionId}"[^>]*>`);
  const match = sectionPattern.exec(interviewHtml);
  if (!match) throw new Error(`Interview guide is missing section #${sectionId}`);
  if (match.index <= previousSectionIndex) {
    throw new Error(`Interview guide section #${sectionId} is out of order`);
  }
  previousSectionIndex = match.index;
}

const interviewIds = [...interviewHtml.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]);
const duplicateInterviewIds = interviewIds.filter(
  (id, index) => interviewIds.indexOf(id) !== index
);
if (duplicateInterviewIds.length) {
  throw new Error(
    `Duplicate interview guide IDs: ${[...new Set(duplicateInterviewIds)].join(", ")}`
  );
}

const interviewIdSet = new Set(interviewIds);
const interviewLocalAnchors = [...interviewHtml.matchAll(/href="#([^"]+)"/g)].map(
  (match) => match[1]
);
const brokenInterviewAnchors = interviewLocalAnchors.filter(
  (anchor) => !interviewIdSet.has(anchor)
);
if (brokenInterviewAnchors.length) {
  throw new Error(
    `Broken interview guide anchors: ${[...new Set(brokenInterviewAnchors)].join(", ")}`
  );
}

const interviewHrefs = [...interviewHtml.matchAll(/href="([^"]+)"/g)].map(
  (match) => match[1]
);
const unsupportedInterviewHrefs = interviewHrefs.filter(
  (href) => !href.startsWith("#") && !href.startsWith("/") && !href.startsWith("https://")
);
if (unsupportedInterviewHrefs.length) {
  throw new Error(
    "Interview guide contains relative links that break on /interview/: " +
    [...new Set(unsupportedInterviewHrefs)].join(", ")
  );
}

const staleInterviewPatterns = [
  /1,429 passed/,
  /1429 passed/,
  /M7 (?:general )?validation (?:is )?(?:complete|accepted)/i,
  /M6 (?:official )?WOMD result (?:is )?accepted/i,
  /the overall winner is/i,
  /best simulator is/i,
  /three causal simulators/i,
  /general nonreactivity detector/i
];
for (const pattern of staleInterviewPatterns) {
  if (pattern.test(interviewHtml)) {
    throw new Error(`Stale interview-guide claim matched ${pattern}`);
  }
}

const metricCards = [
  ...html.matchAll(
    /<article class="metric-card" data-metric-id="([^"]+)" data-version="([^"]+)" data-unit="([^"]+)" data-direction="([^"]+)">([\s\S]*?)<\/article>/g
  )
].map((match) => ({
  id: match[1],
  version: match[2],
  unit: match[3],
  direction: match[4],
  body: match[5]
}));
const expectedMetricCards = [
  {
    id: "position_error_m",
    version: "1.0.0",
    unit: "m",
    question: "How far is the simulated position from the recorded future?",
    required: [
      "Euclidean error for every source-valid non-ego target × post-current future frame",
      "no_eligible_target_frame",
      "Privileged replay is favored by construction",
      "Reactivity, interaction quality, and alternative plausible futures"
    ]
  },
  {
    id: "speed_error_mps",
    version: "1.0.0",
    unit: "m/s",
    question: "How different is scalar speed from the recorded future?",
    required: [
      "Absolute speed-magnitude error for every source-valid non-ego target × post-current future frame",
      "no_eligible_target_frame",
      "Recorded speed-magnitude imitation",
      "Signed direction, position, interaction response"
    ]
  },
  {
    id: "oriented_box_overlap_rate",
    version: "1.0.0",
    unit: "fraction",
    question: "How often does a target box strictly interpenetrate another valid object?",
    required: [
      "One binary flag per source-valid non-ego target × post-current future frame",
      "touching edges are not overlap",
      "Geometric separation",
      "Collision-pair count, severity, safety, and useful motion"
    ]
  },
  {
    id: "waymax_kinematic_infeasibility_rate",
    version: "1.0.1",
    unit: "fraction",
    question: "How often does a vehicle transition violate the pinned inverse-bicycle thresholds?",
    required: [
      "contiguous-valid non-ego vehicle future transition",
      "fixed <code>0.1 s</code> Waymax formula",
      "no_eligible_vehicle_transition",
      "not a physical-time-normalized realism measure"
    ]
  }
];
if (metricCards.length !== expectedMetricCards.length) {
  throw new Error(
    `Presentation has ${metricCards.length} primary metric cards; expected ${expectedMetricCards.length}`
  );
}
for (const [index, expected] of expectedMetricCards.entries()) {
  const card = metricCards[index];
  const required = [
    expected.question,
    ...expected.required,
    "<dt>Components</dt>",
    "<dt>Eligibility</dt>",
    "<dt>Rewards</dt>",
    "<dt>Misses</dt>",
    "paired 128/128 for every contrast",
    "no exclusions or asymmetric missingness",
    "lower is better"
  ];
  const missing = required.filter((fragment) => !card.body.includes(fragment));
  if (
    card.id !== expected.id ||
    card.version !== expected.version ||
    card.unit !== expected.unit ||
    card.direction !== "lower" ||
    missing.length
  ) {
    throw new Error(
      `Primary metric card ${index + 1} changed semantics: expected ${expected.id} ` +
      `v${expected.version} ${expected.unit} lower; missing ${missing.join(", ") || "none"}`
    );
  }
}

const compactResultTable = html.match(
  /<table class="comparison compact-result-table" data-evidence="m5-accepted">([\s\S]*?)<\/table>/
)?.[1];
if (!compactResultTable) throw new Error("Missing compact four-row M5 result summary");
const compactResultBody = compactResultTable.match(/<tbody>([\s\S]*?)<\/tbody>/)?.[1];
const compactRows = [
  ...(compactResultBody ?? "").matchAll(
    /<tr data-metric-id="([^"]+)">([\s\S]*?)<\/tr>/g
  )
].map((match) => ({ id: match[1], body: match[2] }));
const expectedCompactRows = [
  {
    id: "position_error_m",
    ordering: "Replay &lt; CV &lt; IDM",
    boundary: "CV is closer than IDM among the two history-only probes"
  },
  {
    id: "speed_error_mps",
    ordering: "Replay &lt; CV &lt; IDM",
    boundary: "CV is closer than IDM among the two history-only probes"
  },
  {
    id: "oriented_box_overlap_rate",
    ordering: "Replay &lt; IDM &lt; CV",
    boundary: "IDM–CV adjusted band crosses zero"
  },
  {
    id: "waymax_kinematic_infeasibility_rate",
    ordering: "IDM &lt; CV &lt; Replay",
    boundary: "IDM–CV is event-sparse and its adjusted band crosses zero"
  }
];
if (compactRows.length !== expectedCompactRows.length) {
  throw new Error(`Compact M5 summary has ${compactRows.length} rows; expected 4`);
}
for (const [index, expected] of expectedCompactRows.entries()) {
  const row = compactRows[index];
  const visibleCells = row.body.match(/<(?:th|td)\b/g)?.length ?? 0;
  if (
    row.id !== expected.id ||
    visibleCells !== 3 ||
    !row.body.includes(expected.ordering) ||
    !row.body.includes(expected.boundary)
  ) {
    throw new Error(`Compact M5 summary row ${index + 1} changed ordering or boundary`);
  }
}

const constructAuditTable = html.match(
  /<table class="comparison development-matrix" data-evidence="m7-construct-audit">([\s\S]*?)<\/table>/
)?.[1];
if (!constructAuditTable) throw new Error("Missing completed M7 construct-audit matrix");
const constructAuditBody = constructAuditTable.match(/<tbody>([\s\S]*?)<\/tbody>/)?.[1];
const constructAuditRows = [
  ...(constructAuditBody ?? "").matchAll(
    /<tr data-defect="([^"]+)">([\s\S]*?)<\/tr>/g
  )
].map((match) => ({ defect: match[1], body: match[2] }));
const expectedConstructAuditRows = [
  {
    defect: "frozen_agent",
    cells: [
      '<td class="responds">Responds</td>',
      '<td class="no-response">No response</td>',
      '<td class="responds">Responds</td>'
    ]
  },
  {
    defect: "teleportation",
    cells: [
      '<td class="responds">Responds</td>',
      '<td class="no-response">No response</td>',
      '<td class="no-response">No response</td>'
    ]
  },
  {
    defect: "kinematic_spike",
    cells: [
      '<td class="no-response">No response</td>',
      '<td class="no-response">No response</td>',
      '<td class="responds">Responds</td>'
    ]
  },
  {
    defect: "overlap",
    cells: [
      '<td class="responds">Responds</td>',
      '<td class="responds">Responds</td>',
      '<td class="no-response">No response</td>'
    ]
  }
];
if (constructAuditRows.length !== expectedConstructAuditRows.length) {
  throw new Error(`M7 construct-audit matrix has ${constructAuditRows.length} rows; expected 4`);
}
for (const [index, expected] of expectedConstructAuditRows.entries()) {
  const row = constructAuditRows[index];
  const visibleCells = row.body.match(/<(?:th|td)\b/g)?.length ?? 0;
  let cursor = -1;
  const ordered = expected.cells.every((cell) => {
    cursor = row.body.indexOf(cell, cursor + 1);
    return cursor >= 0;
  });
  if (row.defect !== expected.defect || visibleCells !== 4 || !ordered) {
    throw new Error(`M7 construct-audit row ${index + 1} changed the frozen mapping`);
  }
}

const responseCells = constructAuditTable.match(
  /<td class="responds">Responds<\/td>/g
)?.length ?? 0;
const nonresponseCells = constructAuditTable.match(
  /<td class="no-response">No response<\/td>/g
)?.length ?? 0;
if (responseCells !== 6 || nonresponseCells !== 6) {
  throw new Error(
    `M7 construct audit is ${responseCells}/${nonresponseCells}; expected exactly 6/6`
  );
}
if (/Not claimed|Detects|Misses/.test(constructAuditTable)) {
  throw new Error("M7 construct audit contains stale partial-matrix labels");
}

const constructAuditCaveats = [
  "six responses and six non-responses",
  "exactly three hand-built analytic cases at four doses",
  "<code>0.25</code>, <code>0.50</code>, <code>0.75</code>, <code>1.00</code>",
  "v1 freeze changed velocity at the observed current frame",
  "erasing the future deceleration",
  "Freeze v2 preserves history and the current frame",
  "future-only abrupt stop from the next frame",
  "abrupt-stop discontinuity, not generic nonreactivity",
  "not calibrated, source-disjoint, held-out, WOMD-backed, population-level, or general metric validation",
  "<code>speed_error_mps</code> is excluded from this audit",
  "original M7 v1 gates remain unmet",
  "broader validation is optional and separately gated"
];
const missingConstructAuditCaveats = constructAuditCaveats.filter(
  (fragment) => !html.includes(fragment)
);
if (missingConstructAuditCaveats.length) {
  throw new Error(
    `M7 construct-audit caveat is incomplete: ${missingConstructAuditCaveats.join(", ")}`
  );
}

const constructCurveTable = html.match(
  /<table class="comparison construct-curve-table" data-evidence="m7-construct-curves">([\s\S]*?)<\/table>/
)?.[1];
if (!constructCurveTable) throw new Error("Missing exact M7 construct-audit curve table");
const constructCurveBody = constructCurveTable.match(/<tbody>([\s\S]*?)<\/tbody>/)?.[1];
const constructCurveRows = [
  ...(constructCurveBody ?? "").matchAll(
    /<tr data-curve="([^"]+)">([\s\S]*?)<\/tr>/g
  )
].map((match) => ({ curve: match[1], body: match[2] }));
const expectedConstructCurves = [
  ["freeze-position", "position_error_m", ["0.000000", "0.125000", "0.312500", "0.562500", "0.875000"]],
  ["freeze-overlap", "oriented_box_overlap_rate", ["0.000000", "0.000000", "0.000000", "0.000000", "0.000000"]],
  ["freeze-kinematic", "waymax_kinematic_infeasibility_rate", ["0.000000", "0.062500", "0.125000", "0.187500", "0.250000"]],
  ["teleport-position", "position_error_m", ["0.000000", "9.375000", "18.750000", "28.125000", "37.500000"]],
  ["teleport-overlap", "oriented_box_overlap_rate", ["0.000000", "0.000000", "0.000000", "0.000000", "0.000000"]],
  ["teleport-kinematic", "waymax_kinematic_infeasibility_rate", ["0.000000", "0.000000", "0.000000", "0.000000", "0.000000"]],
  ["spike-position", "position_error_m", ["0.000000", "0.000000", "0.000000", "0.000000", "0.000000"]],
  ["spike-overlap", "oriented_box_overlap_rate", ["0.000000", "0.000000", "0.000000", "0.000000", "0.000000"]],
  ["spike-kinematic", "waymax_kinematic_infeasibility_rate", ["0.000000", "0.125000", "0.250000", "0.375000", "0.500000"]],
  ["overlap-position", "position_error_m", ["0.000000", "12.500337", "37.501012", "75.002025", "75.002025"]],
  ["overlap-overlap", "oriented_box_overlap_rate", ["0.000000", "0.500000", "0.750000", "1.000000", "1.000000"]],
  ["overlap-kinematic", "waymax_kinematic_infeasibility_rate", ["0.000000", "0.000000", "0.000000", "0.000000", "0.000000"]]
];
if (constructCurveRows.length !== expectedConstructCurves.length) {
  throw new Error(
    `M7 construct-audit curve table has ${constructCurveRows.length} rows; expected 12`
  );
}
for (const [index, [expectedCurve, expectedMetric, expectedValues]] of
  expectedConstructCurves.entries()) {
  const row = constructCurveRows[index];
  const visibleCells = row.body.match(/<(?:th|td)\b/g)?.length ?? 0;
  let valueCursor = -1;
  const valuesInOrder = expectedValues.every((value) => {
    valueCursor = row.body.indexOf(`>${value}<`, valueCursor + 1);
    return valueCursor >= 0;
  });
  if (
    row.curve !== expectedCurve ||
    visibleCells !== 7 ||
    !row.body.includes(`<code>${expectedMetric}</code>`) ||
    !valuesInOrder
  ) {
    throw new Error(`M7 construct-audit curve row ${index + 1} changed identity or values`);
  }
}
const constructCurveBoundaries = [
  "All values are three-case arithmetic means shown to six decimal places",
  "eligible_components == total_components == 16",
  "These deterministic analytic curves are not uncertainty estimates",
  "The plateau is part of the result",
  "unchanged between doses <code>0.75</code> and <code>1.00</code>",
  "plateau is retained rather than smoothed away"
];
const missingConstructCurveBoundaries = constructCurveBoundaries.filter(
  (fragment) => !html.includes(fragment)
);
if (missingConstructCurveBoundaries.length) {
  throw new Error(
    `M7 construct-audit curve boundary is incomplete: ${missingConstructCurveBoundaries.join(", ")}`
  );
}

const advancedEvidencePatterns = [
  /<details class="evidence-disclosure">[\s\S]*?Open the rollout transition, actor-control boundary, and provenance shape/,
  /<details class="evidence-disclosure">[\s\S]*?Open implementation, cohort, reference, and parity evidence/,
  /<details class="evidence-disclosure">[\s\S]*?Open the complete unchanged 12-cell M5 primary matrix[\s\S]*?<table class="comparison result-table">/,
  /<details class="evidence-disclosure">[\s\S]*?Open contract architecture and milestone-bound implementation status/,
  /<details class="evidence-disclosure">[\s\S]*?Open milestone ledger and local reproduction commands/
];
if (advancedEvidencePatterns.some((pattern) => !pattern.test(html))) {
  throw new Error("Engineering or audit evidence is no longer inside its reviewed disclosure");
}

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
  /data-evidence="m7-development"/,
  /Development evidence only\./,
  /M7 development evidence is not evaluator validation/,
  /Current learning question \/ M7 evaluator validity/,
  /Data-free foundation · M7/,
  /expected v2 matrix/i,
  /outcome-aware v2 proposal requires fresh review/i,
  /proposed v2 study/i,
  /awaits fresh pre-registration/i,
  /before any new held-out execution/i,
  /frozen case remains under generator review/i,
  /not an accepted M7 result/i,
  /M6 not started/,
  /Planned <span>role evidence<\/span>/,
  /updated 2026-07-30/,
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

const hardenedPageHeaders = [
  "content-type",
  "cache-control",
  "content-security-policy",
  "referrer-policy",
  "x-content-type-options",
  "x-frame-options"
];
const interviewRoutes = ["/interview", "/interview/", "/interview-prep.html"];
for (const pathname of interviewRoutes) {
  const response = await worker.fetch(new Request(`https://evalsim.test${pathname}`));
  if (response.status !== 200 || !response.headers.get("content-type")?.startsWith("text/html")) {
    throw new Error(`GET ${pathname} did not return the interview guide`);
  }
  for (const headerName of hardenedPageHeaders) {
    if (response.headers.get(headerName) !== pageResponse.headers.get(headerName)) {
      throw new Error(`GET ${pathname} does not share the hardened page ${headerName}`);
    }
  }
  if (await response.text() !== interviewHtml) {
    throw new Error(`GET ${pathname} returned a stale interview guide`);
  }

  const interviewHeadResponse = await worker.fetch(
    new Request(`https://evalsim.test${pathname}`, { method: "HEAD" })
  );
  if (
    interviewHeadResponse.status !== 200 ||
    (await interviewHeadResponse.arrayBuffer()).byteLength !== 0
  ) {
    throw new Error(`HEAD ${pathname} did not return an empty successful response`);
  }
  for (const headerName of hardenedPageHeaders) {
    if (interviewHeadResponse.headers.get(headerName) !== pageResponse.headers.get(headerName)) {
      throw new Error(`HEAD ${pathname} does not share the hardened page ${headerName}`);
    }
  }
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
  ["disallowed method", new Request("https://evalsim.test/", { method: "POST" }), 405],
  [
    "disallowed interview method",
    new Request("https://evalsim.test/interview", { method: "POST" }),
    405
  ]
];
for (const [name, request, expectedStatus] of cases) {
  const response = await worker.fetch(request);
  if (response.status !== expectedStatus) {
    throw new Error(`${name} returned ${response.status}; expected ${expectedStatus}`);
  }
}

console.log(
  `Site checks passed: ${ids.length + interviewIds.length} unique per-page IDs, ` +
  `${localAnchors.length + interviewLocalAnchors.length} local links, fresh client assets, ` +
  "importable worker, and two-page request matrix"
);
