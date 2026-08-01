import { mkdir, mkdtemp, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const distRoot = resolve(projectRoot, "dist");
const publicRoot = resolve(projectRoot, "public");
const html = await readFile(resolve(projectRoot, "index.html"), "utf8");
const interviewHtml = await readFile(resolve(projectRoot, "interview-prep.html"), "utf8");
const publicEntries = await readdir(publicRoot, { withFileTypes: true });
const unexpectedAssets = publicEntries
  .filter((entry) => entry.name !== "og.png" || !entry.isFile())
  .map((entry) => entry.name);

if (unexpectedAssets.length) {
  throw new Error(
    `Refusing to package unexpected public assets: ${unexpectedAssets.join(", ")}. ` +
    "The deployment allowlist contains only public/og.png."
  );
}

const ogImage = await readFile(resolve(publicRoot, "og.png"));
const ogBase64 = ogImage.toString("base64");
let stagingRoot = await mkdtemp(resolve(projectRoot, ".dist-build-"));

try {
  await mkdir(resolve(stagingRoot, "server"), { recursive: true });
  await mkdir(resolve(stagingRoot, "client"), { recursive: true });
  await writeFile(resolve(stagingRoot, "client", "index.html"), html);
  await writeFile(resolve(stagingRoot, "client", "interview-prep.html"), interviewHtml);
  await writeFile(resolve(stagingRoot, "client", "og.png"), ogImage);

const worker = `const PAGE = ${JSON.stringify(html)};
const INTERVIEW_PAGE = ${JSON.stringify(interviewHtml)};
const OG_BASE64 = ${JSON.stringify(ogBase64)};
const PAGE_HEADERS = {
  "content-type": "text/html; charset=utf-8",
  "cache-control": "public, max-age=300",
  "content-security-policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; object-src 'none'; connect-src 'none'; media-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
  "referrer-policy": "strict-origin-when-cross-origin",
  "x-content-type-options": "nosniff",
  "x-frame-options": "DENY"
};

function decodeBase64(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

export default {
  async fetch(request) {
    const url = new URL(request.url);
    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("Method not allowed", { status: 405, headers: { allow: "GET, HEAD" } });
    }
    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(request.method === "HEAD" ? null : PAGE, { status: 200, headers: PAGE_HEADERS });
    }
    if (
      url.pathname === "/interview" ||
      url.pathname === "/interview/" ||
      url.pathname === "/interview-prep.html"
    ) {
      return new Response(request.method === "HEAD" ? null : INTERVIEW_PAGE, {
        status: 200,
        headers: PAGE_HEADERS
      });
    }
    if (url.pathname === "/og.png" && OG_BASE64) {
      return new Response(request.method === "HEAD" ? null : decodeBase64(OG_BASE64), {
        status: 200,
        headers: {
          "content-type": "image/png",
          "cache-control": "public, max-age=300",
          "x-content-type-options": "nosniff"
        }
      });
    }
    if (url.pathname === "/healthz") {
      return new Response(request.method === "HEAD" ? null : "ok", {
        status: 200,
        headers: { "content-type": "text/plain; charset=utf-8", "cache-control": "no-store" }
      });
    }
    return new Response(request.method === "HEAD" ? null : "Not found", {
      status: 404,
      headers: { "content-type": "text/plain; charset=utf-8" }
    });
  }
};
`;

  await writeFile(resolve(stagingRoot, "server", "index.js"), worker);
  await rm(distRoot, { recursive: true, force: true });
  await rename(stagingRoot, distRoot);
  stagingRoot = "";
} finally {
  if (stagingRoot) await rm(stagingRoot, { recursive: true, force: true });
}

console.log(`Built standalone Sites worker in ${distRoot}`);
