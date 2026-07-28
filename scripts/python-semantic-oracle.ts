const CORPUS_VERSION = 1;
const REQUEST_VERSION = 1;
const RESULT_VERSION = 1;
const REPORT_VERSION = 1;
const MAX_INPUT_BYTES = 4 * 1024 * 1024;
const MAX_FILES = 64;
const MAX_DEPENDENCY_FILES = 256;
const MAX_CASES = 256;
const MAX_DIAGNOSTIC_IDS = 128;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const ID_PATTERN = /^[a-z0-9][a-z0-9_-]{0,127}$/;
const DIAGNOSTIC_ID_PATTERN = /^[A-Za-z][A-Za-z0-9._:-]{0,127}$/;

type JsonRecord = Record<string, unknown>;

export type OracleOutcome = "allow" | "reject" | "unknown";

export interface OracleRequestCase {
  id: string;
  category: string;
  expected_outcome: Exclude<OracleOutcome, "unknown">;
  project: string;
}

export interface PythonOracleRequest {
  version: number;
  corpus_sha256: string;
  python_version: string;
  baseline_project: string;
  dependency_root: string;
  cases: OracleRequestCase[];
}

export interface PythonOracleProvider {
  name: string;
  version: string;
  configuration_sha256: string;
}

export interface PythonOracleResultCase {
  id: string;
  outcome: OracleOutcome;
  diagnostic_ids: string[];
}

export interface PythonOracleResult {
  version: number;
  corpus_sha256: string;
  provider: PythonOracleProvider;
  cases: PythonOracleResultCase[];
}

export interface PythonOracleReport {
  version: number;
  corpus_sha256: string;
  provider: PythonOracleProvider;
  case_count: number;
  agreement_count: number;
  disagreement_count: number;
  unknown_count: number;
  disagreement_ids: string[];
  unknown_ids: string[];
  passed: boolean;
}

interface CorpusFile {
  path: string;
  content: string;
}

interface CorpusCase {
  id: string;
  category: string;
  tool: "write_file" | "replace_file" | "edit_file" | "apply_patch";
  arguments: JsonRecord;
  expected: {
    outcome: "allow" | "reject";
  };
}

interface PythonSemanticCorpus {
  version: number;
  python_version: string;
  files: CorpusFile[];
  dependencies: CorpusFile[];
  cases: CorpusCase[];
}

function record(value: unknown, label: string): JsonRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as JsonRecord;
}

function exactKeys(value: JsonRecord, keys: string[], label: string): void {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (
    actual.length !== expected.length ||
    actual.some((key, index) => key !== expected[index])
  ) {
    throw new Error(`${label} has unknown or missing fields`);
  }
}

function stringField(
  value: JsonRecord,
  key: string,
  label: string,
  max = 256,
): string {
  const field = textField(value, key, label, max);
  if (field.length === 0) {
    throw new Error(`${label}.${key} must be non-empty`);
  }
  return field;
}

function textField(
  value: JsonRecord,
  key: string,
  label: string,
  max = 256,
): string {
  const field = value[key];
  if (typeof field !== "string" || field.length > max || field.includes("\0")) {
    throw new Error(`${label}.${key} must be a bounded string`);
  }
  return field;
}

function arrayField(value: JsonRecord, key: string, label: string): unknown[] {
  const field = value[key];
  if (!Array.isArray(field)) {
    throw new Error(`${label}.${key} must be an array`);
  }
  return field;
}

function versionField(
  value: JsonRecord,
  key: string,
  expected: number,
  label: string,
): void {
  if (value[key] !== expected) {
    throw new Error(`${label}.${key} must be ${expected}`);
  }
}

function logicalPath(value: string, label: string): string {
  if (
    value.startsWith("/") || value.includes("\\") ||
    containsControl(value)
  ) {
    throw new Error(`${label} must be a slash-relative logical path`);
  }
  const components = value.split("/");
  if (
    value.length === 0 ||
    value.length > 512 ||
    components.some((component) =>
      component === "" || component === "." || component === ".."
    )
  ) {
    throw new Error(`${label} contains an empty, dot, or parent component`);
  }
  return value;
}

function containsControl(value: string): boolean {
  return [...value].some((character) => {
    const code = character.charCodeAt(0);
    return code < 32 || code === 127;
  });
}

function safeId(value: string, label: string): string {
  if (!ID_PATTERN.test(value)) {
    throw new Error(`${label} must match ${ID_PATTERN}`);
  }
  return value;
}

async function readBoundedBytes(
  path: string,
  label: string,
): Promise<Uint8Array> {
  const metadata = await Deno.lstat(path);
  if (
    !metadata.isFile || metadata.isSymlink || metadata.size > MAX_INPUT_BYTES
  ) {
    throw new Error(`${label} must be a bounded regular file`);
  }
  const bytes = await Deno.readFile(path);
  if (bytes.length !== metadata.size || bytes.length > MAX_INPUT_BYTES) {
    throw new Error(`${label} changed size while being read`);
  }
  return bytes;
}

async function readBoundedJson(path: string, label: string): Promise<unknown> {
  const bytes = await readBoundedBytes(path, label);
  return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const copy = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(copy).set(bytes);
  const digest = await crypto.subtle.digest("SHA-256", copy);
  return Array.from(
    new Uint8Array(digest),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");
}

function parseCorpus(value: unknown): PythonSemanticCorpus {
  const corpus = record(value, "corpus");
  exactKeys(corpus, [
    "version",
    "python_version",
    "files",
    "dependencies",
    "cases",
  ], "corpus");
  versionField(corpus, "version", CORPUS_VERSION, "corpus");
  const pythonVersion = stringField(corpus, "python_version", "corpus", 32);
  const parseFiles = (key: "files" | "dependencies"): CorpusFile[] => {
    const seen = new Set<string>();
    const candidates = arrayField(corpus, key, "corpus");
    const limit = key === "files" ? MAX_FILES : MAX_DEPENDENCY_FILES;
    if (candidates.length === 0 || candidates.length > limit) {
      throw new Error(`corpus.${key} must contain 1..=${limit} files`);
    }
    return candidates.map((candidate, index) => {
      const file = record(candidate, `corpus.${key}[${index}]`);
      exactKeys(file, ["path", "content"], `corpus.${key}[${index}]`);
      const path = logicalPath(
        stringField(file, "path", `corpus.${key}[${index}]`, 512),
        `${key} path`,
      );
      if (seen.has(path)) {
        throw new Error(`corpus.${key} repeats ${path}`);
      }
      seen.add(path);
      const content = file.content;
      if (typeof content !== "string") {
        throw new Error(`corpus.${key}[${index}].content must be a string`);
      }
      return { path, content };
    });
  };
  const files = parseFiles("files");
  const dependencies = parseFiles("dependencies");
  const rawCases = arrayField(corpus, "cases", "corpus");
  if (rawCases.length === 0 || rawCases.length > MAX_CASES) {
    throw new Error(`corpus.cases must contain 1..=${MAX_CASES} cases`);
  }
  const caseIds = new Set<string>();
  const cases = rawCases.map((candidate, index): CorpusCase => {
    const item = record(candidate, `corpus.cases[${index}]`);
    exactKeys(
      item,
      ["id", "category", "tool", "arguments", "expected"],
      `corpus.cases[${index}]`,
    );
    const id = safeId(
      stringField(item, "id", `corpus.cases[${index}]`, 128),
      "case id",
    );
    if (caseIds.has(id)) {
      throw new Error(`corpus repeats case ${id}`);
    }
    caseIds.add(id);
    const category = safeId(
      stringField(item, "category", `corpus.cases[${index}]`, 128),
      "category",
    );
    const tool = stringField(item, "tool", `corpus.cases[${index}]`, 32);
    if (
      !["write_file", "replace_file", "edit_file", "apply_patch"].includes(tool)
    ) {
      throw new Error(`corpus case ${id} uses unsupported tool ${tool}`);
    }
    const expected = record(item.expected, `corpus.cases[${index}].expected`);
    const expectedKeys = Object.keys(expected);
    if (
      !expectedKeys.includes("outcome") ||
      expectedKeys.some((key) => !["outcome", "diagnostic_codes"].includes(key))
    ) {
      throw new Error(`corpus case ${id} has an invalid expectation`);
    }
    const outcome = stringField(
      expected,
      "outcome",
      `corpus.cases[${index}].expected`,
      16,
    );
    if (outcome !== "allow" && outcome !== "reject") {
      throw new Error(`corpus case ${id} has invalid expected outcome`);
    }
    return {
      id,
      category,
      tool: tool as CorpusCase["tool"],
      arguments: record(item.arguments, `corpus.cases[${index}].arguments`),
      expected: { outcome },
    };
  });
  return {
    version: CORPUS_VERSION,
    python_version: pythonVersion,
    files,
    dependencies,
    cases,
  };
}

async function writeLogicalFile(root: string, file: CorpusFile): Promise<void> {
  const target = `${root}/${logicalPath(file.path, "file path")}`;
  const parent = target.slice(0, target.lastIndexOf("/"));
  await Deno.mkdir(parent, { recursive: true });
  await Deno.writeTextFile(target, file.content, { createNew: true });
}

async function copyCorpusFiles(
  root: string,
  files: CorpusFile[],
): Promise<void> {
  for (const file of files) {
    await writeLogicalFile(root, file);
  }
}

async function assertRegularTree(root: string): Promise<void> {
  let files = 0;
  let bytes = 0;
  const pending = [root];
  while (pending.length > 0) {
    const directory = pending.pop()!;
    for await (const entry of Deno.readDir(directory)) {
      const path = `${directory}/${entry.name}`;
      const metadata = await Deno.lstat(path);
      if (metadata.isSymlink || (!metadata.isFile && !metadata.isDirectory)) {
        throw new Error(
          `materialized oracle tree contains a non-regular entry: ${path}`,
        );
      }
      if (metadata.isDirectory) {
        pending.push(path);
      } else {
        files += 1;
        bytes += metadata.size;
        if (files > 4096 || bytes > 32 * 1024 * 1024) {
          throw new Error(
            "materialized oracle tree exceeds its file or byte bound",
          );
        }
      }
    }
  }
}

async function runGitApply(project: string, patch: string): Promise<void> {
  if (patch.length === 0 || patch.length > MAX_INPUT_BYTES) {
    throw new Error("apply_patch payload must be bounded and non-empty");
  }
  for (
    const args of [["apply", "--check", "--whitespace=nowarn"], [
      "apply",
      "--whitespace=nowarn",
    ]]
  ) {
    const child = new Deno.Command("git", {
      args,
      cwd: project,
      stdin: "piped",
      stdout: "null",
      stderr: "piped",
    }).spawn();
    const writer = child.stdin.getWriter();
    await writer.write(new TextEncoder().encode(patch));
    await writer.close();
    const result = await child.output();
    if (!result.success) {
      const stderr = new TextDecoder().decode(result.stderr).slice(0, 4096);
      throw new Error(
        `git ${args.join(" ")} rejected the canonical corpus patch: ${stderr}`,
      );
    }
  }
}

async function applyMutation(
  project: string,
  mutation: CorpusCase,
): Promise<void> {
  const args = mutation.arguments;
  if (mutation.tool === "apply_patch") {
    exactKeys(args, ["patch"], `case ${mutation.id} arguments`);
    await runGitApply(
      project,
      stringField(
        args,
        "patch",
        `case ${mutation.id} arguments`,
        MAX_INPUT_BYTES,
      ),
    );
    return;
  }

  const path = logicalPath(
    stringField(args, "path", `case ${mutation.id} arguments`, 512),
    "mutation path",
  );
  const target = `${project}/${path}`;
  if (mutation.tool === "write_file") {
    exactKeys(args, ["path", "content"], `case ${mutation.id} arguments`);
    const parent = target.slice(0, target.lastIndexOf("/"));
    await Deno.mkdir(parent, { recursive: true });
    await Deno.writeTextFile(
      target,
      textField(
        args,
        "content",
        `case ${mutation.id} arguments`,
        MAX_INPUT_BYTES,
      ),
      {
        createNew: true,
      },
    );
    return;
  }
  if (mutation.tool === "replace_file") {
    exactKeys(args, ["path", "content"], `case ${mutation.id} arguments`);
    const metadata = await Deno.lstat(target);
    if (!metadata.isFile || metadata.isSymlink) {
      throw new Error(
        `case ${mutation.id} replacement target is not a regular file`,
      );
    }
    await Deno.writeTextFile(
      target,
      textField(
        args,
        "content",
        `case ${mutation.id} arguments`,
        MAX_INPUT_BYTES,
      ),
    );
    return;
  }

  exactKeys(
    args,
    ["path", "old_text", "new_text"],
    `case ${mutation.id} arguments`,
  );
  const oldText = stringField(
    args,
    "old_text",
    `case ${mutation.id} arguments`,
    MAX_INPUT_BYTES,
  );
  const newText = args.new_text;
  if (typeof newText !== "string" || newText.length > MAX_INPUT_BYTES) {
    throw new Error(`case ${mutation.id} new_text must be a bounded string`);
  }
  const base = await Deno.readTextFile(target);
  const first = base.indexOf(oldText);
  if (first < 0 || base.indexOf(oldText, first + oldText.length) >= 0) {
    throw new Error(`case ${mutation.id} old_text must match exactly once`);
  }
  await Deno.writeTextFile(
    target,
    `${base.slice(0, first)}${newText}${base.slice(first + oldText.length)}`,
  );
}

export async function materializeOracleRequest(
  corpusPath: string,
  outputRoot: string,
): Promise<PythonOracleRequest> {
  const corpusBytes = await readBoundedBytes(corpusPath, "corpus");
  const corpus = parseCorpus(
    JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(corpusBytes)),
  );
  await Deno.mkdir(outputRoot);
  const baselineProject = `${outputRoot}/baseline/project`;
  const dependencyRoot = `${outputRoot}/dependencies`;
  await Deno.mkdir(baselineProject, { recursive: true });
  await Deno.mkdir(dependencyRoot, { recursive: true });
  await copyCorpusFiles(baselineProject, corpus.files);
  await copyCorpusFiles(dependencyRoot, corpus.dependencies);

  const requestCases: OracleRequestCase[] = [];
  for (const item of corpus.cases) {
    const project = `${outputRoot}/cases/${item.id}/project`;
    await Deno.mkdir(project, { recursive: true });
    await copyCorpusFiles(project, corpus.files);
    await applyMutation(project, item);
    await assertRegularTree(project);
    requestCases.push({
      id: item.id,
      category: item.category,
      expected_outcome: item.expected.outcome,
      project: `cases/${item.id}/project`,
    });
  }

  const request: PythonOracleRequest = {
    version: REQUEST_VERSION,
    corpus_sha256: await sha256(corpusBytes),
    python_version: corpus.python_version,
    baseline_project: "baseline/project",
    dependency_root: "dependencies",
    cases: requestCases,
  };
  await Deno.writeTextFile(
    `${outputRoot}/oracle-request-v1.json`,
    `${JSON.stringify(request, null, 2)}\n`,
    {
      createNew: true,
    },
  );
  return request;
}

function parseRequest(value: unknown): PythonOracleRequest {
  const request = record(value, "request");
  exactKeys(
    request,
    [
      "version",
      "corpus_sha256",
      "python_version",
      "baseline_project",
      "dependency_root",
      "cases",
    ],
    "request",
  );
  versionField(request, "version", REQUEST_VERSION, "request");
  const corpusSha = stringField(request, "corpus_sha256", "request", 64);
  if (!SHA256_PATTERN.test(corpusSha)) {
    throw new Error("request corpus_sha256 must be lowercase SHA-256");
  }
  const seen = new Set<string>();
  const rawCases = arrayField(request, "cases", "request");
  if (rawCases.length === 0 || rawCases.length > MAX_CASES) {
    throw new Error(`request.cases must contain 1..=${MAX_CASES} cases`);
  }
  const cases = rawCases.map(
    (candidate, index): OracleRequestCase => {
      const item = record(candidate, `request.cases[${index}]`);
      exactKeys(
        item,
        ["id", "category", "expected_outcome", "project"],
        `request.cases[${index}]`,
      );
      const id = safeId(
        stringField(item, "id", `request.cases[${index}]`, 128),
        "request case id",
      );
      if (seen.has(id)) throw new Error(`request repeats case ${id}`);
      seen.add(id);
      const outcome = stringField(
        item,
        "expected_outcome",
        `request.cases[${index}]`,
        16,
      );
      if (outcome !== "allow" && outcome !== "reject") {
        throw new Error(`request case ${id} has invalid outcome`);
      }
      return {
        id,
        category: safeId(
          stringField(item, "category", `request.cases[${index}]`, 128),
          "category",
        ),
        expected_outcome: outcome,
        project: logicalPath(
          stringField(item, "project", `request.cases[${index}]`, 512),
          "case project",
        ),
      };
    },
  );
  return {
    version: REQUEST_VERSION,
    corpus_sha256: corpusSha,
    python_version: stringField(request, "python_version", "request", 32),
    baseline_project: logicalPath(
      stringField(request, "baseline_project", "request", 512),
      "baseline project",
    ),
    dependency_root: logicalPath(
      stringField(request, "dependency_root", "request", 512),
      "dependency root",
    ),
    cases,
  };
}

function parseResult(value: unknown): PythonOracleResult {
  const result = record(value, "result");
  exactKeys(
    result,
    ["version", "corpus_sha256", "provider", "cases"],
    "result",
  );
  versionField(result, "version", RESULT_VERSION, "result");
  const corpusSha = stringField(result, "corpus_sha256", "result", 64);
  if (!SHA256_PATTERN.test(corpusSha)) {
    throw new Error("result corpus_sha256 must be lowercase SHA-256");
  }
  const rawProvider = record(result.provider, "result.provider");
  exactKeys(
    rawProvider,
    ["name", "version", "configuration_sha256"],
    "result.provider",
  );
  const configurationSha = stringField(
    rawProvider,
    "configuration_sha256",
    "result.provider",
    64,
  );
  if (!SHA256_PATTERN.test(configurationSha)) {
    throw new Error(
      "result provider configuration_sha256 must be lowercase SHA-256",
    );
  }
  const provider: PythonOracleProvider = {
    name: safeId(
      stringField(rawProvider, "name", "result.provider", 128),
      "provider name",
    ),
    version: stringField(rawProvider, "version", "result.provider", 128),
    configuration_sha256: configurationSha,
  };
  if (containsControl(provider.version)) {
    throw new Error("result provider version must be a single-line identity");
  }
  const seen = new Set<string>();
  const rawCases = arrayField(result, "cases", "result");
  if (rawCases.length === 0 || rawCases.length > MAX_CASES) {
    throw new Error(`result.cases must contain 1..=${MAX_CASES} cases`);
  }
  const cases = rawCases.map(
    (candidate, index): PythonOracleResultCase => {
      const item = record(candidate, `result.cases[${index}]`);
      exactKeys(
        item,
        ["id", "outcome", "diagnostic_ids"],
        `result.cases[${index}]`,
      );
      const id = safeId(
        stringField(item, "id", `result.cases[${index}]`, 128),
        "result case id",
      );
      if (seen.has(id)) throw new Error(`result repeats case ${id}`);
      seen.add(id);
      const outcome = stringField(
        item,
        "outcome",
        `result.cases[${index}]`,
        16,
      );
      if (
        outcome !== "allow" && outcome !== "reject" && outcome !== "unknown"
      ) {
        throw new Error(`result case ${id} has invalid outcome`);
      }
      const rawDiagnosticIds = arrayField(
        item,
        "diagnostic_ids",
        `result case ${id}`,
      );
      if (rawDiagnosticIds.length > MAX_DIAGNOSTIC_IDS) {
        throw new Error(`result case ${id} has too many diagnostics`);
      }
      const diagnosticIds = rawDiagnosticIds.map(
        (diagnostic, diagnosticIndex) => {
          if (
            typeof diagnostic !== "string" ||
            !DIAGNOSTIC_ID_PATTERN.test(diagnostic)
          ) {
            throw new Error(
              `result case ${id} diagnostic ${diagnosticIndex} is invalid`,
            );
          }
          return diagnostic;
        },
      );
      const sorted = [...new Set(diagnosticIds)].sort();
      if (
        sorted.length !== diagnosticIds.length ||
        sorted.some((diagnostic, diagnosticIndex) =>
          diagnostic !== diagnosticIds[diagnosticIndex]
        )
      ) {
        throw new Error(
          `result case ${id} diagnostic_ids must be sorted and unique`,
        );
      }
      return { id, outcome, diagnostic_ids: diagnosticIds };
    },
  );
  return { version: RESULT_VERSION, corpus_sha256: corpusSha, provider, cases };
}

export function compareOracleResult(
  requestValue: unknown,
  resultValue: unknown,
): PythonOracleReport {
  const request = parseRequest(requestValue);
  const result = parseResult(resultValue);
  if (request.corpus_sha256 !== result.corpus_sha256) {
    throw new Error(
      "oracle result is not bound to the exact materialized corpus",
    );
  }
  if (
    result.cases.length !== request.cases.length ||
    request.cases.some((item, index) => result.cases[index]?.id !== item.id)
  ) {
    throw new Error(
      "oracle result must contain one verdict for every request case in request order",
    );
  }
  let agreementCount = 0;
  const disagreementIds: string[] = [];
  const unknownIds: string[] = [];
  for (const [index, item] of request.cases.entries()) {
    const candidate = result.cases[index];
    if (candidate.outcome === "unknown") unknownIds.push(item.id);
    else if (candidate.outcome === item.expected_outcome) agreementCount += 1;
    else disagreementIds.push(item.id);
  }
  return {
    version: REPORT_VERSION,
    corpus_sha256: request.corpus_sha256,
    provider: result.provider,
    case_count: request.cases.length,
    agreement_count: agreementCount,
    disagreement_count: disagreementIds.length,
    unknown_count: unknownIds.length,
    disagreement_ids: disagreementIds,
    unknown_ids: unknownIds,
    passed: disagreementIds.length === 0 && unknownIds.length === 0,
  };
}

function option(args: string[], name: string): string {
  const index = args.indexOf(name);
  if (
    index < 0 || index + 1 >= args.length || args[index + 1].startsWith("--")
  ) {
    throw new Error(`missing ${name}`);
  }
  if (args.indexOf(name, index + 1) >= 0) throw new Error(`repeated ${name}`);
  return args[index + 1];
}

async function main(args: string[]): Promise<void> {
  const command = args[0];
  if (command === "materialize") {
    if (args.length !== 5) {
      throw new Error(
        "usage: materialize --corpus <path> --output <new-directory>",
      );
    }
    const request = await materializeOracleRequest(
      option(args, "--corpus"),
      option(args, "--output"),
    );
    console.log(JSON.stringify(request, null, 2));
    return;
  }
  if (command === "compare") {
    const requireAgreement = args.includes("--require-agreement");
    const expectedLength = requireAgreement ? 6 : 5;
    if (args.length !== expectedLength) {
      throw new Error(
        "usage: compare --request <path> --result <path> [--require-agreement]",
      );
    }
    const request = await readBoundedJson(
      option(args, "--request"),
      "oracle request",
    );
    const result = await readBoundedJson(
      option(args, "--result"),
      "oracle result",
    );
    const report = compareOracleResult(request, result);
    console.log(JSON.stringify(report, null, 2));
    if (requireAgreement && !report.passed) Deno.exit(2);
    return;
  }
  throw new Error("expected materialize or compare command");
}

if (import.meta.main) {
  try {
    await main(Deno.args);
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    Deno.exit(1);
  }
}
