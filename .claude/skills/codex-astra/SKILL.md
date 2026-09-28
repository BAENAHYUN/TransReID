---
name: codex-astra
description: Delegate a bounded review or implementation task to local Codex GPT-6 Astra and inspect its result.
argument-hint: "[review|fix] <task and target files>"
disable-model-invocation: true
---

# Delegate to Codex Astra

User request: $ARGUMENTS

This project command delegates to a separate **Codex CLI process**, fixed to
`gpt-6-astra`. Claude remains the coordinator. This is not a Claude-native model
selection. Codex does not inherit this conversation: supply the relevant context.

## Environment

Run from the TransReID project root on **local Windows**, including Claude Code
using Git Bash. The wrapper uses the existing Windows Codex login and discovers
the Codex desktop app's executable even when `codex` is absent from PATH.
Do not copy authentication files or ask for an API key.

If this session runs in a remote Linux VM/container without access to Windows
processes, stop and explain that this command requires local Windows Claude Code.
Do not try to bypass network policies. For WSL, use Windows Python via interop
only if available and use Windows-compatible paths; otherwise report the limitation.

## Workflow

1. Interpret the optional first word: `review` (default) or `fix`.
   Use `fix` only when the user explicitly requests it. If the request is empty,
   ask what work to delegate. Review may inspect code but cannot edit project files.
2. Read the minimum project context needed to specify a bounded task. For `fix`,
   identify the exact allowed files and capture their current diff/content so the
   resulting edits can be assessed without confusing existing user edits.
   Do not edit the same files while Codex is working.
3. With the Write tool, create a new UTF-8 text file at
   `codex_out/requests/<unique-timestamp-or-uuid>.txt`. Include:
   - The user's objective and acceptance criteria.
   - Mode, allowed target files, useful reference files, and relevant prior decisions.
   - Current symptoms, exact error messages, and what was already tried.
   - Project constraints: preserve Hybrid-C grouping, person invariants, named-vector
     dimensions/model contracts, and unrelated user changes.
   - Do not access or modify live Qdrant, run ingestion or GPU builds, delete/recreate
     collections, use destructive pipeline flags, push, commit, or deploy.
   - In review mode: no edits and no executing project scripts/tests; read and analyze.
   - In fix mode: edit only the listed files; use bounded relevant tests, with no live DB.
   - Return Korean findings with severity and file/line references, or changed files,
     validation results and remaining limitations. Distinguish evidence from assumptions.
   Do not include secrets. Do not put the task text directly into a shell command.
4. Invoke the wrapper with an argument **file**, not interpolated task text:

   ```bash
   .venv/Scripts/python.exe .claude/skills/codex-astra/invoke.py --mode review --prompt-file codex_out/requests/UNIQUE.txt
   ```

   Replace `review` with `fix` only for an explicit fix request. Replace `UNIQUE.txt`
   with the actual file created above. Quote paths when necessary. Keep normal
   Claude Code permission handling; do not disable approvals or sandboxing.
   Allow this to run to completion, following the shell tool's background task if
   necessary. Do not start duplicate runs because the task is still running.
5. Read the printed `status.json` and `result.md` paths. Success requires
   `status == "completed"`, `exit_code == 0`, and a nonempty result. On failure,
   inspect `stderr.log` and `events.jsonl`; report the real error. Do not silently
   change models, claim success, or rerun a partially completed fix automatically.
6. Independently check findings against code. For `fix`, inspect the actual diff
   and relevant test results; Codex's summary alone is not validation. Report what
   Astra did, your verification, and the result path to the user in Korean.

## Checks and outputs

Check the local CLI without invoking a model:

```bash
.venv/Scripts/python.exe .claude/skills/codex-astra/invoke.py --check
```

Each invocation creates `codex_out/astra/<unique-run>/` with `prompt.txt`,
`result.md`, `events.jsonl`, `stderr.log`, and `status.json`. Runs are independent;
include context again for a follow-up. Never use another run's output as this run's result.

Examples:

```text
/codex-astra review search/image_search.py의 reranker none 경로를 검토해줘
/codex-astra review ingest/build_db.py의 resume 날짜 경계 처리를 확인해줘
/codex-astra fix search/image_search.py의 확인된 정렬 버그만 수정하고 검증해줘
```
