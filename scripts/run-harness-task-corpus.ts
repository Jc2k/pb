export interface CorpusSeedFile {
  path: string;
  content: string;
}

export interface CorpusLimits {
  max_steps: number;
  max_tokens: number;
}

export interface CorpusCase {
  id: string;
  category: string;
  task: string;
  seed_files: CorpusSeedFile[];
  resume_files: CorpusSeedFile[];
  contract: Record<string, unknown>;
  limits: CorpusLimits;
}

export interface TaskCorpus {
  version: 1;
  cases: CorpusCase[];
}

const CASE_ID = /^[a-z][a-z0-9_]{2,63}$/;

function requireObject(value: unknown, label: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function requireInteger(
  value: unknown,
  label: string,
  minimum: number,
  maximum: number,
): number {
  if (
    typeof value !== "number" || !Number.isInteger(value) || value < minimum ||
    value > maximum
  ) {
    throw new Error(
      `${label} must be an integer from ${minimum} to ${maximum}`,
    );
  }
  return value;
}

export function validateRelativePath(path: string, label: string): string {
  const normalized = path.replaceAll("\\", "/");
  if (
    normalized.length === 0 || normalized.startsWith("/") ||
    normalized.split("/").some((part) =>
      part.length === 0 || part === "." || part === ".."
    )
  ) {
    throw new Error(`${label} must stay beneath the corpus workspace: ${path}`);
  }
  return normalized;
}

function stringArray(value: unknown, label: string): string[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`);
  return value.map((item, index) => requireString(item, `${label}[${index}]`));
}

function validateFiles(value: unknown, label: string): CorpusSeedFile[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`);
  const paths = new Set<string>();
  return value.map((rawFile, index) => {
    const file = requireObject(rawFile, `${label}[${index}]`);
    const path = validateRelativePath(
      requireString(file.path, `${label}[${index}].path`),
      `${label}[${index}].path`,
    );
    if (paths.has(path)) throw new Error(`${label} repeats path: ${path}`);
    paths.add(path);
    if (typeof file.content !== "string") {
      throw new Error(`${label}[${index}].content must be a string`);
    }
    return { path, content: file.content };
  });
}

export function validateCorpus(value: unknown): TaskCorpus {
  const root = requireObject(value, "corpus");
  if (root.version !== 1) throw new Error("corpus version must be 1");
  if (!Array.isArray(root.cases)) {
    throw new Error("corpus cases must be an array");
  }
  if (root.cases.length < 10 || root.cases.length > 20) {
    throw new Error("corpus must contain 10 to 20 cases");
  }

  const ids = new Set<string>();
  const cases = root.cases.map((rawCase, caseIndex): CorpusCase => {
    const item = requireObject(rawCase, `cases[${caseIndex}]`);
    const id = requireString(item.id, `cases[${caseIndex}].id`);
    if (!CASE_ID.test(id)) throw new Error(`invalid corpus case id: ${id}`);
    if (ids.has(id)) throw new Error(`duplicate corpus case id: ${id}`);
    ids.add(id);

    const seedFiles = validateFiles(item.seed_files, `${id}.seed_files`);
    const resumeFiles = validateFiles(
      item.resume_files ?? [],
      `${id}.resume_files`,
    );

    const contract = requireObject(item.contract, `${id}.contract`);
    if (contract.version !== 1) {
      throw new Error(`${id}.contract.version must be 1`);
    }
    const allowedPaths = stringArray(
      contract.allowed_paths ?? [],
      `${id}.contract.allowed_paths`,
    )
      .map((path, index) =>
        validateRelativePath(path, `${id}.contract.allowed_paths[${index}]`)
      );
    if (new Set(allowedPaths).size !== allowedPaths.length) {
      throw new Error(`${id}.contract.allowed_paths contains duplicates`);
    }
    const workUnitGuidance = requireObject(
      contract.work_unit_guidance ?? {},
      `${id}.contract.work_unit_guidance`,
    );
    if (Object.keys(workUnitGuidance).length > 64) {
      throw new Error(`${id}.contract.work_unit_guidance has more than 64 entries`);
    }
    let guidanceBytes = 0;
    for (const [rawPath, rawGuidance] of Object.entries(workUnitGuidance)) {
      const path = validateRelativePath(
        rawPath,
        `${id}.contract.work_unit_guidance path`,
      );
      if (allowedPaths.length > 0 && !allowedPaths.includes(path)) {
        throw new Error(
          `${id}.contract.work_unit_guidance path is not allowed: ${path}`,
        );
      }
      const guidance = requireString(
        rawGuidance,
        `${id}.contract.work_unit_guidance[${path}]`,
      );
      const bytes = new TextEncoder().encode(guidance.trim()).length;
      if (bytes > 512) {
        throw new Error(
          `${id}.contract.work_unit_guidance[${path}] exceeds 512 bytes`,
        );
      }
      guidanceBytes += bytes;
    }
    if (guidanceBytes > 4096) {
      throw new Error(`${id}.contract.work_unit_guidance exceeds 4096 bytes`);
    }
    if (!Array.isArray(contract.checks)) {
      throw new Error(`${id}.contract.checks must be an array`);
    }
    const checkIds = new Set<string>();
    for (const [checkIndex, rawCheck] of contract.checks.entries()) {
      const check = requireObject(
        rawCheck,
        `${id}.contract.checks[${checkIndex}]`,
      );
      const checkId = requireString(
        check.id,
        `${id}.contract.checks[${checkIndex}].id`,
      );
      if (checkIds.has(checkId)) {
        throw new Error(`${id} repeats check id: ${checkId}`);
      }
      checkIds.add(checkId);
      requireString(
        check.command,
        `${id}.contract.checks[${checkIndex}].command`,
      );
    }
    const review = requireObject(
      contract.review ?? {},
      `${id}.contract.review`,
    );
    for (
      const [index, path] of stringArray(
        review.read_paths ?? [],
        `${id}.contract.review.read_paths`,
      ).entries()
    ) {
      validateRelativePath(path, `${id}.contract.review.read_paths[${index}]`);
    }
    for (
      const checkId of stringArray(
        review.check_ids ?? [],
        `${id}.contract.review.check_ids`,
      )
    ) {
      if (!checkIds.has(checkId)) {
        throw new Error(`${id} review names unknown check: ${checkId}`);
      }
    }

    const limits = requireObject(item.limits, `${id}.limits`);
    return {
      id,
      category: requireString(item.category, `${id}.category`),
      task: requireString(item.task, `${id}.task`),
      seed_files: seedFiles,
      resume_files: resumeFiles,
      contract,
      limits: {
        max_steps: requireInteger(
          limits.max_steps,
          `${id}.limits.max_steps`,
          1,
          12,
        ),
        max_tokens: requireInteger(
          limits.max_tokens,
          `${id}.limits.max_tokens`,
          256,
          4096,
        ),
      },
    };
  });

  return { version: 1, cases };
}

const encoder = new TextEncoder();

function compareUtf8(left: string, right: string): number {
  const leftBytes = encoder.encode(left);
  const rightBytes = encoder.encode(right);
  const sharedLength = Math.min(leftBytes.byteLength, rightBytes.byteLength);
  for (let index = 0; index < sharedLength; index++) {
    if (leftBytes[index] !== rightBytes[index]) {
      return leftBytes[index] - rightBytes[index];
    }
  }
  return leftBytes.byteLength - rightBytes.byteLength;
}

function joinBytes(parts: Uint8Array[]): Uint8Array {
  const joined = new Uint8Array(
    parts.reduce((total, part) => total + part.byteLength, 0),
  );
  let offset = 0;
  for (const part of parts) {
    joined.set(part, offset);
    offset += part.byteLength;
  }
  return joined;
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const input = new Uint8Array(bytes.byteLength);
  input.set(bytes);
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", input.buffer),
  );
  return Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join(
    "",
  );
}

function littleEndianLength(length: number): Uint8Array {
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(length), true);
  return bytes;
}

async function writeTaskBaseline(
  scratchRoot: string,
  workspace: string,
  files: CorpusSeedFile[],
): Promise<void> {
  const sorted = [...files].sort((left, right) =>
    compareUtf8(left.path, right.path)
  );
  const fingerprintParts: Uint8Array[] = [];
  const paths: Record<string, { kind: string; fingerprint: string }> = {};
  for (const file of sorted) {
    const path = encoder.encode(file.path);
    const kind = encoder.encode("file");
    const content = encoder.encode(file.content);
    fingerprintParts.push(
      littleEndianLength(path.byteLength),
      path,
      kind,
      content,
    );
    paths[file.path] = {
      kind: "file",
      fingerprint: await sha256Hex(joinBytes([kind, content])),
    };
  }
  const contentFingerprint = await sha256Hex(joinBytes(fingerprintParts));
  const head = await commandText(
    "git",
    ["rev-parse", "--verify", "HEAD"],
    workspace,
  );
  const zero = new Uint8Array([0]);
  const id = await sha256Hex(joinBytes([
    encoder.encode(head),
    zero,
    zero,
    encoder.encode(contentFingerprint),
  ]));
  await Deno.writeTextFile(
    `${scratchRoot}/task-baseline.json`,
    `${
      JSON.stringify(
        {
          id,
          head,
          status: { porcelain: "", dirty_paths: [] },
          content: { fingerprint: contentFingerprint, paths },
        },
        null,
        2,
      )
    }\n`,
  );
}

export async function loadCorpus(path: string): Promise<TaskCorpus> {
  return validateCorpus(JSON.parse(await Deno.readTextFile(path)));
}

async function runCommand(
  command: string,
  args: string[],
  cwd: string,
): Promise<void> {
  const output = await new Deno.Command(command, { args, cwd }).output();
  if (!output.success) {
    const diagnostic = new TextDecoder().decode(output.stderr).trim();
    throw new Error(`${command} ${args.join(" ")} failed: ${diagnostic}`);
  }
}

async function commandText(
  command: string,
  args: string[],
  cwd: string,
): Promise<string> {
  const output = await new Deno.Command(command, { args, cwd }).output();
  if (!output.success) {
    const diagnostic = new TextDecoder().decode(output.stderr).trim();
    throw new Error(`${command} ${args.join(" ")} failed: ${diagnostic}`);
  }
  return new TextDecoder().decode(output.stdout).trim();
}

export async function prepareCorpusCase(
  corpusCase: CorpusCase,
  scratchRoot: string,
): Promise<{ scratch_root: string; workspace: string; contract: string }> {
  try {
    await Deno.lstat(scratchRoot);
    throw new Error(`scratch path already exists: ${scratchRoot}`);
  } catch (error) {
    if (!(error instanceof Deno.errors.NotFound)) throw error;
  }

  const workspace = `${scratchRoot}/workspace`;
  await Deno.mkdir(workspace, { recursive: true });
  await runCommand("git", ["init", "-b", "main"], workspace);
  await runCommand(
    "git",
    ["config", "user.name", "pb corpus harness"],
    workspace,
  );
  await runCommand(
    "git",
    ["config", "user.email", "corpus@pb.local"],
    workspace,
  );
  for (const seed of corpusCase.seed_files) {
    const destination = `${workspace}/${seed.path}`;
    const parent = destination.slice(0, destination.lastIndexOf("/"));
    await Deno.mkdir(parent, { recursive: true });
    await Deno.writeTextFile(destination, seed.content);
  }
  await runCommand("git", ["add", "--all"], workspace);
  await runCommand(
    "git",
    ["commit", "--allow-empty", "-m", "chore: initialize corpus workspace"],
    workspace,
  );
  if (corpusCase.resume_files.length > 0) {
    await writeTaskBaseline(scratchRoot, workspace, corpusCase.seed_files);
    for (const resumed of corpusCase.resume_files) {
      const destination = `${workspace}/${resumed.path}`;
      const parent = destination.slice(0, destination.lastIndexOf("/"));
      await Deno.mkdir(parent, { recursive: true });
      await Deno.writeTextFile(destination, resumed.content);
    }
  }

  const contract = `${scratchRoot}/contract.json`;
  await Deno.writeTextFile(
    contract,
    `${JSON.stringify(corpusCase.contract, null, 2)}\n`,
  );
  await Deno.writeTextFile(
    `${scratchRoot}/task.txt`,
    `${corpusCase.task.trim()}\n`,
  );
  await Deno.writeTextFile(
    `${scratchRoot}/corpus-case.json`,
    `${
      JSON.stringify(
        {
          version: 1,
          case_id: corpusCase.id,
          category: corpusCase.category,
          resumed: corpusCase.resume_files.length > 0,
        },
        null,
        2,
      )
    }\n`,
  );
  return { scratch_root: scratchRoot, workspace, contract };
}

interface CliOptions {
  manifest: string;
  caseId?: string;
  scratchDir?: string;
  binary: string;
  list: boolean;
  prepareOnly: boolean;
}

function parseCli(args: string[]): CliOptions {
  const options: CliOptions = {
    manifest: "fixtures/harness-task-completion/corpus.json",
    binary: "target/aarch64-apple-darwin/release/pb",
    list: false,
    prepareOnly: false,
  };
  for (let index = 0; index < args.length; index++) {
    const argument = args[index];
    if (argument === "--list") options.list = true;
    else if (argument === "--prepare-only") options.prepareOnly = true;
    else if (
      ["--manifest", "--case", "--scratch-dir", "--binary"].includes(argument)
    ) {
      const value = args[++index];
      if (!value) throw new Error(`${argument} requires a value`);
      if (argument === "--manifest") options.manifest = value;
      else if (argument === "--case") options.caseId = value;
      else if (argument === "--scratch-dir") options.scratchDir = value;
      else options.binary = value;
    } else throw new Error(`unknown argument: ${argument}`);
  }
  return options;
}

async function main(): Promise<void> {
  const options = parseCli(Deno.args);
  const corpus = await loadCorpus(options.manifest);
  if (options.list) {
    for (const item of corpus.cases) {
      console.log(`${item.id}\t${item.category}\t${item.task}`);
    }
    return;
  }
  if (!options.caseId || !options.scratchDir) {
    throw new Error("running a case requires --case and --scratch-dir");
  }
  const corpusCase = corpus.cases.find((item) => item.id === options.caseId);
  if (!corpusCase) throw new Error(`unknown corpus case: ${options.caseId}`);
  const prepared = await prepareCorpusCase(corpusCase, options.scratchDir);
  if (options.prepareOnly) {
    console.log(JSON.stringify({ case_id: corpusCase.id, ...prepared }));
    return;
  }

  const command = new Deno.Command(options.binary, {
    args: [
      "harness",
      "agent",
      "--scratch-dir",
      prepared.scratch_root,
      "--contract",
      prepared.contract,
      "--max-steps",
      String(corpusCase.limits.max_steps),
      "--max-tokens",
      String(corpusCase.limits.max_tokens),
      "--temperature",
      "0",
      "--top-k",
      "1",
      "--seed",
      "0",
      corpusCase.task,
    ],
    stdin: "inherit",
    stdout: "inherit",
    stderr: "inherit",
  });
  const status = await command.spawn().status;
  if (!status.success) Deno.exit(status.code);
}

if (import.meta.main) {
  try {
    await main();
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    Deno.exit(2);
  }
}
