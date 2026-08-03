import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import test from "node:test";

const harness = path.resolve(
  ".superpowers/sdd/2026-07-31-paddleocr-labeler-service/live_browser_e2e.cjs",
);
const expectedNames = Array.from(
  { length: 10 },
  (_, index) => `2_${14 + index}.png`,
);
const { createTemporaryWorkspace, openWorkspace } = createRequire(import.meta.url)(harness);

test("application shell uses the visible dynamic viewport without fixed header subtraction", () => {
  const css = fs.readFileSync(
    path.resolve("ocr_labeler/static/styles.css"),
    "utf8",
  );
  const bodyRule = css.match(/body\s*\{([^}]*)\}/s)?.[1] ?? "";
  const workspaceRule = css.match(/\.workspace\s*\{([^}]*)\}/s)?.[1] ?? "";

  assert.match(bodyRule, /display:\s*grid\s*;/);
  assert.match(bodyRule, /grid-template-rows:\s*auto minmax\(0,\s*1fr\)\s*;/);
  assert.match(bodyRule, /height:\s*100dvh\s*;/);
  assert.match(bodyRule, /overflow-y:\s*hidden\s*;/);
  assert.match(workspaceRule, /height:\s*auto\s*;/);
  assert.match(workspaceRule, /min-height:\s*0\s*;/);
  assert.doesNotMatch(workspaceRule, /calc\(100vh\s*-\s*64px\)/);
});

function makeFixture(prefix, names = expectedNames) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  for (const name of names) {
    fs.writeFileSync(path.join(directory, name), `fixture:${name}`);
  }
  return directory;
}

function existsWithoutFollowing(target) {
  try {
    fs.lstatSync(target);
    return true;
  } catch (error) {
    if (error?.code === "ENOENT") return false;
    throw error;
  }
}

test("temporary browser workspace copies only expected images and cleans its exact path", () => {
  const source = makeFixture("labeler-e2e-source-");
  const arbitraryWorkspace = makeFixture("labeler-e2e-do-not-write-", ["sentinel"]);
  const program = `
    const assert = require("node:assert/strict");
    const fs = require("node:fs");
    const path = require("node:path");
    const harness = require(${JSON.stringify(harness)});
    const lease = harness.createTemporaryWorkspace(${JSON.stringify(source)});
    assert.notEqual(
      fs.realpathSync(lease.path),
      fs.realpathSync(${JSON.stringify(arbitraryWorkspace)}),
    );
    assert.deepEqual(fs.readdirSync(lease.path).sort(), ${JSON.stringify(expectedNames)});
    for (const name of harness.EXPECTED_IMAGE_NAMES) {
      assert.equal(
        fs.readFileSync(path.join(lease.path, name), "utf8"),
        "fixture:" + name,
      );
    }
    const temporaryPath = lease.path;
    lease.cleanup();
    assert.equal(fs.existsSync(temporaryPath), false);
    assert.equal(
      fs.readFileSync(path.join(${JSON.stringify(arbitraryWorkspace)}, "sentinel"), "utf8"),
      "fixture:sentinel",
    );
  `;

  try {
    const result = spawnSync(process.execPath, ["-e", program], {
      encoding: "utf8",
      env: {
        ...process.env,
        LABELER_E2E_WORKSPACE: arbitraryWorkspace,
      },
    });
    assert.equal(result.status, 0, result.stderr || result.stdout);
  } finally {
    fs.rmSync(source, { recursive: true, force: true });
    fs.rmSync(arbitraryWorkspace, { recursive: true, force: true });
  }
});

test("openWorkspace uses its explicit workspace argument", async () => {
  const actions = [];
  const page = {
    locator(selector) {
      return {
        async fill(value) { actions.push(["fill", selector, value]); },
        async click() { actions.push(["click", selector]); },
      };
    },
    async waitForFunction(_predicate, root, options) {
      actions.push(["wait", root, options.timeout]);
    },
  };

  await openWorkspace(page, "/tmp/explicit-workspace");

  assert.deepEqual(actions, [
    ["fill", "#folder-path", "/tmp/explicit-workspace"],
    ["click", "#open-folder"],
    ["wait", "/tmp/explicit-workspace", 15_000],
  ]);
});

test("missing source image fails before browser launch and cannot target legacy workspace", () => {
  const source = makeFixture("labeler-e2e-incomplete-", expectedNames.slice(0, -1));
  const arbitraryWorkspace = makeFixture("labeler-e2e-protected-", ["sentinel"]);

  try {
    const result = spawnSync(process.execPath, [harness], {
      encoding: "utf8",
      env: {
        ...process.env,
        LABELER_E2E_SOURCE: source,
        LABELER_E2E_WORKSPACE: arbitraryWorkspace,
      },
    });

    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /missing source image: 2_23\.png/);
    assert.equal(
      fs.readFileSync(path.join(arbitraryWorkspace, "sentinel"), "utf8"),
      "fixture:sentinel",
    );
    assert.deepEqual(fs.readdirSync(arbitraryWorkspace), ["sentinel"]);
  } finally {
    fs.rmSync(source, { recursive: true, force: true });
    fs.rmSync(arbitraryWorkspace, { recursive: true, force: true });
  }
});

test("cleanup refuses a replacement directory and preserves both directory identities", () => {
  const source = makeFixture("labeler-e2e-race-source-");
  const lease = createTemporaryWorkspace(source);
  const renamedOriginal = `${lease.path}.renamed-original`;

  try {
    fs.renameSync(lease.path, renamedOriginal);
    fs.mkdirSync(lease.path);
    fs.writeFileSync(path.join(lease.path, "replacement-sentinel"), "preserve me");

    assert.throws(
      () => lease.cleanup(),
      /refusing to clean replaced workspace/,
    );
    assert.equal(
      fs.readFileSync(path.join(lease.path, "replacement-sentinel"), "utf8"),
      "preserve me",
    );
    assert.deepEqual(fs.readdirSync(renamedOriginal).sort(), expectedNames);
  } finally {
    if (existsWithoutFollowing(lease.path)) {
      fs.rmSync(lease.path, { recursive: true, force: true });
    }
    if (existsWithoutFollowing(renamedOriginal)) {
      fs.rmSync(renamedOriginal, { recursive: true, force: true });
    }
    fs.rmSync(source, { recursive: true, force: true });
  }
});

test("cleanup unlinks a dangling replacement symlink without following its target", () => {
  const source = makeFixture("labeler-e2e-symlink-source-");
  const lease = createTemporaryWorkspace(source);
  const renamedOriginal = `${lease.path}.renamed-original`;
  const absentTarget = `${lease.path}.must-stay-absent`;

  try {
    fs.renameSync(lease.path, renamedOriginal);
    fs.symlinkSync(absentTarget, lease.path);

    lease.cleanup();

    assert.equal(existsWithoutFollowing(lease.path), false);
    assert.equal(existsWithoutFollowing(absentTarget), false);
    assert.deepEqual(fs.readdirSync(renamedOriginal).sort(), expectedNames);
  } finally {
    if (existsWithoutFollowing(lease.path)) fs.unlinkSync(lease.path);
    if (existsWithoutFollowing(renamedOriginal)) {
      fs.rmSync(renamedOriginal, { recursive: true, force: true });
    }
    if (existsWithoutFollowing(absentTarget)) {
      fs.rmSync(absentTarget, { recursive: true, force: true });
    }
    fs.rmSync(source, { recursive: true, force: true });
  }
});
