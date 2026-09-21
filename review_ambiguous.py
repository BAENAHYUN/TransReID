from __future__ import annotations

"""
Local browser-based reviewer for ambiguous dedup pairs.

Reads:
    data/crops/dedup_report.json

Writes:
    data/crops/manual_review.json

Purpose:
    Review only pairs that automatic Hybrid-C dedup marked as "ambiguous".
    This reviewer records a human relation decision only. It never deletes crop JPGs,
    never removes detection points, and never modifies dedup_report.json.

Hybrid-C decisions:
    duplicate
        -> reviewer believes A/B are the same physical object.
           apply_manual_review.py keeps BOTH detections and assigns the same
           duplicate_group_id. No semantic point deletion/merge occurs.

    separate
        -> reviewer believes A/B are different physical objects.
           Both detections remain independent.

    keep_ambiguous
        -> reviewer is still uncertain.
           Both detections remain independent and are linked only by
           ambiguous_group_id after apply_manual_review.py.

Important:
    The decision string "duplicate" describes the semantic relation only.
    Under Hybrid C it does NOT mean "delete one point" or "merge into a survivor".

Run:
    python review_ambiguous.py

Optional:
    python review_ambiguous.py ^
        --report ".\\data\\crops\\dedup_report.json" ^
        --output ".\\data\\crops\\manual_review.json" ^
        --port 8765
"""

import argparse
import hashlib
import json
import mimetypes
import os
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = SCRIPT_ROOT / "data" / "crops" / "dedup_report.json"
DEFAULT_OUTPUT = SCRIPT_ROOT / "data" / "crops" / "manual_review.json"


HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hybrid C Ambiguous Dedup Review</title>
<style>
:root {
  color-scheme: dark;
  --bg:#111318; --panel:#1a1e25; --muted:#9ba3af; --text:#f4f6f8;
  --border:#343b46; --accent:#7aa2ff; --good:#64d98b; --warn:#f0c36b;
  --selected:#26344f;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }
header {
  position:sticky; top:0; z-index:2; display:flex; gap:16px; align-items:center;
  justify-content:space-between; padding:12px 18px; background:#151920;
  border-bottom:1px solid var(--border);
}
header .title { font-size:18px; font-weight:700; }
header .progress { color:var(--muted); font-size:14px; }
main { max-width:1500px; margin:0 auto; padding:18px; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.card { background:var(--panel); border:1px solid var(--border); border-radius:12px; overflow:hidden; }
.card h2 { margin:0; padding:10px 14px; font-size:16px; border-bottom:1px solid var(--border); }
.image-wrap {
  height:52vh; min-height:300px; display:flex; align-items:center; justify-content:center;
  background:#0b0d11; padding:10px;
}
.image-wrap img { max-width:100%; max-height:100%; object-fit:contain; }
.meta {
  padding:12px 14px; display:grid; grid-template-columns:max-content 1fr;
  gap:6px 12px; font-size:14px;
}
.meta .k { color:var(--muted); }
.metrics {
  margin-top:16px; background:var(--panel); border:1px solid var(--border);
  border-radius:12px; padding:14px;
  display:grid; grid-template-columns:repeat(6,minmax(110px,1fr)); gap:10px;
}
.metric { background:#151920; border:1px solid var(--border); border-radius:8px; padding:10px; }
.metric .k { color:var(--muted); font-size:12px; }
.metric .v { font-size:16px; font-weight:700; margin-top:3px; word-break:break-word; }
.reason { margin-top:12px; color:var(--muted); font-size:13px; }
.actions {
  position:sticky; bottom:0; margin-top:16px; padding:14px; background:#151920ee;
  backdrop-filter:blur(8px); border:1px solid var(--border); border-radius:12px;
}
.buttons { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }
button {
  padding:14px 10px; border-radius:9px; border:1px solid var(--border);
  background:#232832; color:var(--text); font-size:15px; font-weight:700; cursor:pointer;
}
button:hover { filter:brightness(1.12); }
button.dup { border-color:#6e5ad7; }
button.sep { border-color:#3aa66a; }
button.amb { border-color:#c8993d; }
button.active { background:var(--selected); outline:2px solid var(--accent); }
.nav { display:flex; justify-content:space-between; gap:10px; margin-top:10px; }
.nav button { flex:1; padding:9px; font-size:13px; }
.statusline {
  display:flex; justify-content:space-between; align-items:center; gap:12px;
  margin-top:10px; color:var(--muted); font-size:13px;
}
.badge { padding:4px 8px; border-radius:999px; border:1px solid var(--border); color:var(--text); }
.badge.pending { color:var(--warn); }
.badge.duplicate { color:#b5a7ff; }
.badge.separate { color:var(--good); }
.badge.keep_ambiguous { color:var(--warn); }
kbd {
  border:1px solid var(--border); border-bottom-width:2px; border-radius:4px;
  padding:1px 5px; font-family:inherit; font-size:12px; background:#20252d;
}
@media (max-width:900px) {
  .grid { grid-template-columns:1fr; }
  .image-wrap { height:38vh; }
  .metrics { grid-template-columns:repeat(2,1fr); }
}
</style>
</head>
<body>
<header>
  <div class="title">Hybrid C · Ambiguous Dedup Review</div>
  <div class="progress" id="progress">loading...</div>
</header>
<main>
  <div class="grid">
    <section class="card">
      <h2>A</h2>
      <div class="image-wrap"><img id="imgA" alt="crop A"></div>
      <div class="meta" id="metaA"></div>
    </section>
    <section class="card">
      <h2>B</h2>
      <div class="image-wrap"><img id="imgB" alt="crop B"></div>
      <div class="meta" id="metaB"></div>
    </section>
  </div>

  <section class="metrics">
    <div class="metric"><div class="k">Tier</div><div class="v" id="tier">-</div></div>
    <div class="metric"><div class="k">IoU</div><div class="v" id="iou">-</div></div>
    <div class="metric"><div class="k">IoM</div><div class="v" id="iom">-</div></div>
    <div class="metric"><div class="k">Area ratio</div><div class="v" id="area">-</div></div>
    <div class="metric"><div class="k">Center dist.</div><div class="v" id="center">-</div></div>
    <div class="metric"><div class="k">DINO cosine</div><div class="v" id="dino">-</div></div>
  </section>
  <div class="reason" id="reason"></div>
  <div class="reason">
    Hybrid C: <b>D</b>는 한 detection을 지우는 선택이 아니라,
    두 detection을 같은 <code>duplicate_group_id</code>로 묶는 선택입니다.
  </div>

  <section class="actions">
    <div class="buttons">
      <button class="dup" id="btnDup" onclick="decide('duplicate')">D · Duplicate → Group</button>
      <button class="sep" id="btnSep" onclick="decide('separate')">S · Separate</button>
      <button class="amb" id="btnAmb" onclick="decide('keep_ambiguous')">A · Keep Ambiguous</button>
    </div>
    <div class="nav">
      <button onclick="go(-1)">← Previous</button>
      <button onclick="jumpPending()">Next Pending</button>
      <button onclick="go(1)">Next →</button>
    </div>
    <div class="statusline">
      <div>
        <kbd>D</kbd> duplicate→group (no deletion) · <kbd>S</kbd> separate · <kbd>A</kbd> ambiguous ·
        <kbd>←</kbd>/<kbd>→</kbd> navigate
      </div>
      <div class="badge pending" id="statusBadge">pending</div>
    </div>
  </section>
</main>

<script>
let state = null;
let idx = 0;

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'
  }[c]));
}
function fmt(v, digits=6) {
  if (v === null || v === undefined || v === '') return '-';
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(digits).replace(/0+$/,'').replace(/\.$/,'') : esc(v);
}
function metaHtml(x) {
  const bbox = Array.isArray(x.bbox) ? x.bbox.join(', ') : '-';
  return `
    <div class="k">class</div><div>${esc(x.class_name)}</div>
    <div class="k">confidence</div><div>${fmt(x.confidence)}</div>
    <div class="k">detection_id</div><div>${esc(x.detection_id)}</div>
    <div class="k">bbox</div><div>[${esc(bbox)}]</div>
    <div class="k">crop</div><div>${esc(x.crop_path)}</div>
  `;
}
function current() { return state.pairs[idx]; }

function render() {
  if (!state || state.pairs.length === 0) {
    document.body.innerHTML = '<main><h2>No ambiguous pairs found.</h2></main>';
    return;
  }
  idx = Math.max(0, Math.min(idx, state.pairs.length - 1));
  const p = current();
  const g = p.geometry || {};
  const r = state.reviews[p.pair_id];

  document.getElementById('imgA').src = `/image/${idx}/a?ts=${Date.now()}`;
  document.getElementById('imgB').src = `/image/${idx}/b?ts=${Date.now()}`;
  document.getElementById('metaA').innerHTML = metaHtml(p.a || {});
  document.getElementById('metaB').innerHTML = metaHtml(p.b || {});
  document.getElementById('tier').textContent = g.tier ?? '-';
  document.getElementById('iou').textContent = fmt(g.iou);
  document.getElementById('iom').textContent = fmt(g.iom);
  document.getElementById('area').textContent = fmt(g.area_ratio);
  document.getElementById('center').textContent = fmt(g.center_distance);
  document.getElementById('dino').textContent = fmt(p.dino_similarity);
  document.getElementById('reason').textContent =
    `auto reason: ${p.reason ?? '-'} · pair_id: ${p.pair_id}`;

  const done = Object.keys(state.reviews).length;
  document.getElementById('progress').textContent =
    `${idx + 1} / ${state.pairs.length} · reviewed ${done} · pending ${state.pairs.length - done}`;

  const badge = document.getElementById('statusBadge');
  const decision = r?.decision ?? 'pending';
  badge.textContent = decision;
  badge.className = `badge ${decision}`;

  for (const id of ['btnDup','btnSep','btnAmb']) {
    document.getElementById(id).classList.remove('active');
  }
  if (decision === 'duplicate') document.getElementById('btnDup').classList.add('active');
  if (decision === 'separate') document.getElementById('btnSep').classList.add('active');
  if (decision === 'keep_ambiguous') document.getElementById('btnAmb').classList.add('active');
}

async function load() {
  const res = await fetch('/api/state');
  state = await res.json();
  const firstPending = state.pairs.findIndex(p => !state.reviews[p.pair_id]);
  idx = firstPending >= 0 ? firstPending : 0;
  render();
}

async function decide(decision) {
  const p = current();
  const res = await fetch('/api/review', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({pair_id:p.pair_id, decision})
  });
  const body = await res.json();
  if (!res.ok) {
    alert(body.error || 'save failed');
    return;
  }
  state.reviews[p.pair_id] = body.review;
  render();

  const next = state.pairs.findIndex((x, i) => i > idx && !state.reviews[x.pair_id]);
  if (next >= 0) { idx = next; render(); }
}

function go(delta) {
  idx = Math.max(0, Math.min(state.pairs.length - 1, idx + delta));
  render();
}
function jumpPending() {
  if (!state) return;
  for (let step = 1; step <= state.pairs.length; step++) {
    const j = (idx + step) % state.pairs.length;
    if (!state.reviews[state.pairs[j].pair_id]) {
      idx = j; render(); return;
    }
  }
  alert('모든 ambiguous pair 검토가 완료되었습니다.');
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'd' || e.key === 'D') decide('duplicate');
  else if (e.key === 's' || e.key === 'S') decide('separate');
  else if (e.key === 'a' || e.key === 'A') decide('keep_ambiguous');
  else if (e.key === 'ArrowLeft') go(-1);
  else if (e.key === 'ArrowRight') go(1);
});

load();
</script>
</body>
</html>
"""


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pair_id(pair: dict[str, Any]) -> str:
    a = str((pair.get("a") or {}).get("detection_id") or "")
    b = str((pair.get("b") or {}).get("detection_id") or "")
    image_id = str(pair.get("image_id") or "")
    key = image_id + "|" + "|".join(sorted([a, b]))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


def _normalize_pair(pair: dict[str, Any]) -> dict[str, Any]:
    out = dict(pair)
    out["pair_id"] = _pair_id(pair)
    return out


class ReviewStore:
    def __init__(self, report_path: Path, output_path: Path, project_root: Path):
        self.report_path = report_path
        self.output_path = output_path
        self.project_root = project_root
        self.lock = threading.Lock()
        self.report_sha256 = _sha256(report_path)

        report = _read_json(report_path)

        # Hybrid-C safety gate. Old A-style reports physically removed semantic
        # duplicates, so require a report produced by the Hybrid-C dedup stage.
        mode = str(report.get("mode") or "").strip()
        expected_mode = "hybrid_c_hard_exact_soft_semantic"
        if mode != expected_mode:
            raise RuntimeError(
                "dedup_report.json is not a Hybrid-C report. "
                f"expected mode={expected_mode!r}, actual={mode!r}. "
                "Run the Hybrid-C dedup_objects.py again before manual review."
            )

        removed_count = int(report.get("removed_count", 0) or 0)
        exact_removed = int(report.get("exact_detection_id_removed", 0) or 0)
        same_removed = int(report.get("same_class_removed", 0) or 0)
        cross_removed = int(report.get("cross_class_removed", 0) or 0)

        if removed_count != exact_removed or same_removed != 0 or cross_removed != 0:
            raise RuntimeError(
                "Hybrid-C invariant violated in dedup_report.json: "
                "only exact detection_id duplicates may be physically removed. "
                f"removed_count={removed_count}, exact={exact_removed}, "
                f"same_class_removed={same_removed}, cross_class_removed={cross_removed}"
            )

        raw_pairs = report.get("ambiguous_pairs", [])
        if not isinstance(raw_pairs, list):
            raise TypeError('dedup_report.json: "ambiguous_pairs" must be a list.')

        self.pairs = [_normalize_pair(x) for x in raw_pairs if isinstance(x, dict)]
        self.pair_by_id = {p["pair_id"]: p for p in self.pairs}

        self.reviews: dict[str, dict[str, Any]] = {}
        if output_path.is_file():
            existing = _read_json(output_path)
            if not isinstance(existing, dict):
                raise RuntimeError(
                    "Existing manual_review.json is malformed. Rename/delete it "
                    "before starting a new Hybrid-C review."
                )

            format_version = int(existing.get("format_version", 0) or 0)
            dedup_mode = str(existing.get("dedup_mode") or "").strip()
            source_hash = str(existing.get("source_report_sha256") or "").strip()
            semantics = existing.get("review_semantics") or {}

            if (
                format_version < 3
                or dedup_mode != expected_mode
                or source_hash != self.report_sha256
                or not isinstance(semantics, dict)
                or semantics.get("semantic_point_deletion") is not False
            ):
                raise RuntimeError(
                    "Existing manual_review.json is stale or from an older review "
                    "contract. Refusing to reuse decisions silently. Rename/delete "
                    "the old manual_review.json and review the current report again. "
                    f"format_version={format_version}, dedup_mode={dedup_mode!r}, "
                    f"report_hash_match={source_hash == self.report_sha256}"
                )

            raw_reviews = existing.get("reviews", {})
            if not isinstance(raw_reviews, dict):
                raise RuntimeError("manual_review.json: reviews must be an object")
            for pid, review in raw_reviews.items():
                if pid in self.pair_by_id and isinstance(review, dict):
                    self.reviews[pid] = review

    def resolve_crop(self, pair_index: int, side: str) -> Path:
        if pair_index < 0 or pair_index >= len(self.pairs):
            raise IndexError("pair index out of range")
        side = side.lower()
        if side not in {"a", "b"}:
            raise ValueError("side must be a or b")

        record = self.pairs[pair_index].get(side) or {}
        raw = record.get("crop_path")
        if raw is None or not str(raw).strip():
            raise FileNotFoundError("crop_path missing")

        p = Path(str(raw))
        if p.is_file():
            return p

        if not p.is_absolute():
            alt = (self.project_root / p).resolve()
            if alt.is_file():
                return alt

        raise FileNotFoundError(str(p))

    def save_review(self, pair_id: str, decision: str) -> dict[str, Any]:
        if pair_id not in self.pair_by_id:
            raise KeyError("unknown pair_id")

        allowed = {"duplicate", "separate", "keep_ambiguous"}
        if decision not in allowed:
            raise ValueError(f"decision must be one of {sorted(allowed)}")

        pair = self.pair_by_id[pair_id]
        a = pair.get("a") or {}
        b = pair.get("b") or {}

        review = {
            "decision": decision,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "image_id": pair.get("image_id"),
            "a_detection_id": a.get("detection_id"),
            "b_detection_id": b.get("detection_id"),
        }

        with self.lock:
            self.reviews[pair_id] = review
            payload = {
                "format_version": 3,
                "dedup_mode": "hybrid_c_hard_exact_soft_semantic",
                "review_semantics": {
                    "duplicate": "preserve_both_and_assign_duplicate_group_id",
                    "separate": "preserve_both_as_independent_points",
                    "keep_ambiguous": "preserve_both_and_assign_ambiguous_group_id",
                    "semantic_point_deletion": False,
                },
                "source_report": str(self.report_path),
                "source_report_sha256": self.report_sha256,
                "total_ambiguous_pairs": len(self.pairs),
                "reviewed_count": len(self.reviews),
                "pending_count": len(self.pairs) - len(self.reviews),
                "reviews": self.reviews,
            }
            _atomic_write_json(self.output_path, payload)

        return review

    def state_payload(self) -> dict[str, Any]:
        return {
            "mode": "hybrid_c_hard_exact_soft_semantic",
            "duplicate_action": "group_only_no_point_deletion",
            "total": len(self.pairs),
            "pairs": self.pairs,
            "reviews": self.reviews,
            "output": str(self.output_path),
            "source_report_sha256": self.report_sha256,
        }


def make_handler(store: ReviewStore):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HybridCAmbiguousReview/2.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_json(self, obj: Any, status: int = 200) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_text(self, text: str, content_type: str, status: int = 200) -> None:
            raw = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            path = urlparse(self.path).path

            if path == "/":
                self._send_text(HTML, "text/html; charset=utf-8")
                return

            if path == "/api/state":
                self._send_json(store.state_payload())
                return

            parts = [p for p in path.split("/") if p]
            if len(parts) == 3 and parts[0] == "image":
                try:
                    idx = int(parts[1])
                    side = parts[2].lower()
                    image_path = store.resolve_crop(idx, side)
                    content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
                    data = image_path.read_bytes()

                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(data)
                except Exception as exc:
                    self._send_text(
                        f"Image unavailable: {exc}",
                        "text/plain; charset=utf-8",
                        status=404,
                    )
                return

            self._send_text("Not found", "text/plain; charset=utf-8", status=404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/review":
                self._send_json({"error": "Not found"}, status=404)
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 65536:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                pair_id = str(body.get("pair_id") or "")
                decision = str(body.get("decision") or "")
                review = store.save_review(pair_id, decision)
                self._send_json({"ok": True, "review": review})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Review Hybrid-C ambiguous pairs. Duplicate means grouping, never point deletion."
    )
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--root",
        default=str(SCRIPT_ROOT),
        help="Project root used to resolve relative crop paths.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the default browser.",
    )
    args = parser.parse_args()

    report_path = Path(args.report).resolve()
    output_path = Path(args.output).resolve()
    project_root = Path(args.root).resolve()

    if not report_path.is_file():
        raise FileNotFoundError(f"dedup report not found: {report_path}")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")

    store = ReviewStore(report_path, output_path, project_root)

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(store),
    )
    url = f"http://{args.host}:{args.port}/"

    print()
    print("=" * 72)
    print("HYBRID C AMBIGUOUS DEDUP REVIEW")
    print("=" * 72)
    print(f"report          : {report_path}")
    print(f"ambiguous pairs : {len(store.pairs):,}")
    print(f"already reviewed: {len(store.reviews):,}")
    print(f"output          : {output_path}")
    print(f"url             : {url}")
    print()
    print("Keys: D=duplicate->group(no deletion), S=separate, A=keep ambiguous, arrows=navigate")
    print("Hybrid C: manual duplicate decisions preserve both points and only create a group.")
    print("Press Ctrl+C to stop. Reviews are saved immediately.")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
