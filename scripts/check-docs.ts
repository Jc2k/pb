import { dirname, extname, join, normalize } from "node:path/posix";

const siteRoot = Deno.args[0] ?? "site";
const sitePrefix = "/pb/";
const files = new Set<string>();
const html = new Map<string, string>();

async function collect(directory: string, relative = ""): Promise<void> {
  for await (const entry of Deno.readDir(directory)) {
    const path = join(relative, entry.name);
    if (entry.isDirectory) {
      await collect(join(directory, entry.name), path);
    } else if (entry.isFile) {
      files.add(path);
      if (extname(path) === ".html") {
        html.set(path, await Deno.readTextFile(join(siteRoot, path)));
      }
    }
  }
}

function decode(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function localReference(value: string): boolean {
  return !/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(value);
}

function targetPath(source: string, reference: string): string {
  const path = decode(reference.split("#", 1)[0].split("?", 1)[0]);
  if (!path) return source;

  let target: string;
  if (path.startsWith(sitePrefix)) {
    target = path.slice(sitePrefix.length);
  } else if (path.startsWith("/")) {
    target = path.slice(1);
  } else {
    target = normalize(join(dirname(source), path));
  }

  if (!target) target = "index.html";
  if (target.endsWith("/")) target = `${target}index.html`;
  if (!extname(target) && files.has(`${target}.html`)) {
    target = `${target}.html`;
  }
  return target;
}

function targetFragment(reference: string): string | undefined {
  const marker = reference.indexOf("#");
  if (marker < 0) return undefined;
  const fragment = reference.slice(marker + 1).split("?", 1)[0];
  return fragment ? decode(fragment) : undefined;
}

await collect(siteRoot);

if (html.size === 0 || !html.has("index.html")) {
  throw new Error(`no rendered documentation found in ${siteRoot}`);
}

const failures: string[] = [];
for (const [source, contents] of html) {
  for (
    const required of [
      "viewport-fit=cover",
      'name="apple-mobile-web-app-capable"',
      'rel="manifest"',
    ]
  ) {
    if (!contents.includes(required)) {
      failures.push(`${source}: missing required head content ${required}`);
    }
  }

  const references = contents.matchAll(/(?:href|src)=(?:"([^"]*)"|'([^']*)')/g);
  for (const match of references) {
    const reference = (match[1] ?? match[2]).replaceAll("&amp;", "&");
    if (!reference || !localReference(reference)) continue;

    const target = targetPath(source, reference);
    if (target.startsWith("../") || !files.has(target)) {
      failures.push(`${source}: ${reference} points to missing ${target}`);
      continue;
    }

    const fragment = targetFragment(reference);
    if (!fragment || extname(target) !== ".html") continue;
    const targetHtml = html.get(target) ??
      await Deno.readTextFile(join(siteRoot, target));
    const ids = new Set(
      [...targetHtml.matchAll(/\s(?:id|name)=(?:"([^"]+)"|'([^']+)')/g)]
        .map((id) => id[1] ?? id[2]),
    );
    if (!ids.has(fragment)) {
      failures.push(
        `${source}: ${reference} points to missing fragment #${fragment}`,
      );
    }
  }
}

const manifestPath = join(siteRoot, "manifest.webmanifest");
if (!files.has("manifest.webmanifest")) {
  failures.push("manifest.webmanifest was not copied to the rendered site");
} else {
  const manifest = JSON.parse(await Deno.readTextFile(manifestPath));
  for (const icon of manifest.icons ?? []) {
    const iconPath = String(icon.src).replace(/^\.\//, "");
    if (!files.has(iconPath)) {
      failures.push(`manifest icon is missing: ${iconPath}`);
    }
  }
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  Deno.exit(1);
}

console.log(`checked ${html.size} pages and ${files.size} rendered files`);
