"""Windows-local Claude Code -> Codex Astra bridge (Python standard library only)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4

MODEL = "gpt-6-astra"
ROOT = Path(__file__).resolve().parents[3]


def find_codex() -> Path:
    # Prefer a native CLI; .ps1/.cmd shims would need shell-dependent quoting.
    candidate = shutil.which("codex.exe")
    if candidate:
        return Path(candidate)
    app_root = Path(os.environ.get("LOCALAPPDATA", "")) / "OpenAI/Codex/bin"
    candidates = sorted(app_root.glob("*/codex.exe"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise RuntimeError("Codex executable not found. Install/sign in to Codex on local Windows.")


def write_status(path: Path, data: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW, check=False)
        if process.poll() is None:
            process.kill()
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("review", "fix"), default="review")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Use local Windows Python, not a remote VM/container Python.")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    codex = find_codex()
    flags = subprocess.CREATE_NO_WINDOW
    if args.check:
        print(f"Codex: {codex}\nModel: {MODEL}\nProject: {ROOT}")
        for command in ([str(codex), "--version"], [str(codex), "login", "status"]):
            result = subprocess.run(command, capture_output=True, encoding="utf-8",
                                    errors="replace", timeout=30, creationflags=flags)
            print(result.stdout.strip())
            print(result.stderr.strip(), file=sys.stderr)
            if result.returncode:
                return result.returncode
        return 0
    if args.prompt_file is None:
        parser.error("--prompt-file is required unless --check is used")
    source = args.prompt_file.resolve()
    task = source.read_text(encoding="utf-8-sig")
    if not task.strip():
        parser.error("Prompt file is empty")

    sandbox = "read-only" if args.mode == "review" else "workspace-write"
    constraints = (
        "You are Codex Astra, receiving a delegated task from Claude Code. "
        "Follow repository instructions. Do not delegate again. Report in Korean.\n"
        "Do not access or modify live databases, run ingestion/GPU pipelines, delete "
        "or recreate collections, use --recreate/--fresh/--recreate-person/--recreate-object, "
        "change embedding dimensions/model contracts, commit, push, or deploy.\n"
    )
    if args.mode == "review":
        constraints += "REVIEW ONLY: read/analyze files. No edits. Do not execute project scripts or tests.\n"
    else:
        constraints += (
            "FIX MODE: edit only task-listed files. Preserve unrelated changes. "
            "If allowed files are not specified, explain the missing scope without editing. "
            "Run only relevant bounded tests that do not use live databases.\n"
        )
    prompt = constraints + "\nDelegated task:\n" + task
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:10]
    run_dir = ROOT / "codex_out" / "astra" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    output = run_dir / "result.md"
    status_path = run_dir / "status.json"
    command = [str(codex), "exec", "--model", MODEL,
               "--config", 'model_reasoning_effort="high"',
               "--config", 'approval_policy="never"',
               "--sandbox", sandbox, "--cd", str(ROOT),
               "--skip-git-repo-check", "--ephemeral", "--color", "never",
               "--json", "--output-last-message", str(output), "-"]
    status = {"status": "running", "model": MODEL, "mode": args.mode,
              "sandbox": sandbox, "codex": str(codex), "project": str(ROOT),
              "started_at": datetime.now(timezone.utc).isoformat(),
              "exit_code": None, "result": str(output), "prompt_source": str(source)}
    write_status(status_path, status)
    print(f"Run: {run_dir}\nModel: {MODEL}\nMode: {args.mode}", flush=True)
    process = None
    code = 1
    try:
        with (run_dir / "events.jsonl").open("wb") as stdout, (run_dir / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE,
                                       stdout=stdout, stderr=stderr, creationflags=flags)
            status["pid"] = process.pid
            write_status(status_path, status)
            process.communicate(input=prompt.encode("utf-8"), timeout=args.timeout_seconds)
        code = process.returncode
        if code == 0 and output.is_file() and output.read_text(encoding="utf-8").strip():
            status["status"] = "completed"
        else:
            status["status"] = "failed"
            status["error"] = "Codex failed or produced no final response; inspect stderr.log and events.jsonl."
            code = code or 1
    except subprocess.TimeoutExpired:
        status.update(status="timed_out", error="Timed out; review any partial changes before retrying.")
        code = 124
    except KeyboardInterrupt:
        status.update(status="cancelled", error="Interrupted; review any partial changes before retrying.")
        code = 130
    except Exception as exc:
        status.update(status="failed", error=str(exc))
    finally:
        if process is not None:
            stop_process(process)
        status["exit_code"] = code
        status["codex_exit_code"] = process.returncode if process is not None else None
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_status(status_path, status)
        print(f"Status: {status_path}\nResult: {output}\nExit code: {code}", flush=True)
    return code


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Codex Astra bridge: {exc}", file=sys.stderr)
        sys.exit(1)
