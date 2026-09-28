"""eval/gt_sheet.py — 사람 라벨링 시트(HTML) 공용 도우미 (P6 정답 확보).

세 시트(추적 구간 · 객체 재출현 쌍 · Qwen 판정)가 같은 방식으로 동작한다:
  - 썸네일은 base64 로 HTML 안에 넣는다 (파일 하나만 옮기면 다른 PC 에서도 열린다).
  - 입력 요소에 data-item / data-field 를 달면 자동 수집·자동 저장(localStorage)·JSON 내보내기·불러오기가 된다.
  - 항목마다 "검토" 체크(data-field="reviewed")가 있다. 어떤 필드든 손대면 자동으로 체크되고, 체크된 항목만 완료로 센다.
    제안값(기본값)을 그대로 두고 저장한 항목은 검토된 것이 아니다 — 평가 스크립트는 reviewed 가 아닌 항목을 사람 정답으로 쓰지 않는다.
  - manifest: 시트를 만든 항목 집합의 해시. localStorage 키·labels.json·평가 입력이 같은 manifest 를 공유해야 하며,
    시트를 다시 만들면(항목이 바뀌면) 옛 라벨이 조용히 붙지 않는다.
  - 내보낸 labels.json = {"kind", "meta"(manifest 포함), "labeler", "exported_at", "labels": {item_id: {field: value, "reviewed": bool}}}.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIELDS = ("gt_id", "status", "split_frame", "split_gt_id", "note", "verdict", "relevant")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def esc(text: Any) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def display_name(path: Any) -> str:
    """시트에 절대경로를 노출하지 않는다 — 파일명만."""
    if not path:
        return ""
    return Path(str(path)).name


def load_json(path: Any, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return default
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_text(path: Any, text: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return p


def write_json(path: Any, data: Any) -> Path:
    return write_text(path, json.dumps(data, ensure_ascii=False, indent=1, default=_json_default))


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def manifest_of(kind: str, item_ids: Iterable[Any], extra: Any = None) -> str:
    """항목 집합(+부가 정보)의 해시 12자. 항목 id 순서와 무관."""
    payload = {"kind": kind, "items": sorted(str(i) for i in item_ids), "extra": extra}
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default).encode("utf-8")).hexdigest()[:12]


def file_sha1(path: Any) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha1(p.read_bytes()).hexdigest()[:12]


def resolve_path(value: Any, root: Path = ROOT) -> Optional[Path]:
    """절대경로면 그대로, 상대경로면 프로젝트 루트 기준. 다른 PC 의 절대경로면 data/ · outputs/ 뒤쪽만 붙여 복구."""
    if not value:
        return None
    raw = Path(str(value))
    if raw.is_file():
        return raw
    if not raw.is_absolute():
        cand = root / raw
        if cand.is_file():
            return cand
    parts = [p for p in str(value).replace("\\", "/").split("/")]
    for anchor in ("data", "outputs", "eval"):
        if anchor in parts:
            idx = parts.index(anchor)
            cand = root.joinpath(*parts[idx:])
            if cand.is_file():
                return cand
    return None


# ---------------------------------------------------------------- 썸네일
def crop_with_margin(frame: np.ndarray, bbox: Sequence[float], margin: float = 0.2, min_side: int = 8) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    x1 -= bw * margin
    x2 += bw * margin
    y1 -= bh * margin
    y2 += bh * margin
    xa, ya = max(int(x1), 0), max(int(y1), 0)
    xb, yb = min(int(round(x2)), w), min(int(round(y2)), h)
    if xb - xa < min_side:
        xb = min(xa + min_side, w)
    if yb - ya < min_side:
        yb = min(ya + min_side, h)
    return frame[ya:yb, xa:xb]


def thumb_b64_from_array(img_bgr: np.ndarray, height: int = 160, quality: int = 80) -> str:
    """cv2 BGR 배열 → data URI (JPEG). 실패하면 ''."""
    try:
        import cv2
        if img_bgr is None or img_bgr.size == 0:
            return ""
        h, w = img_bgr.shape[:2]
        if h != height:
            scale = height / float(h)
            img_bgr = cv2.resize(img_bgr, (max(int(round(w * scale)), 1), height), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return ""


def thumb_b64_from_file(path: Any, height: int = 160, quality: int = 80) -> str:
    """이미지 파일 → data URI (JPEG). 없거나 못 읽으면 ''."""
    p = resolve_path(path)
    if p is None:
        return ""
    try:
        from PIL import Image
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            if h != height and h > 0:
                im = im.resize((max(int(round(w * height / h)), 1), height))
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:
        return ""


def img_tag(data_uri: str, alt: str = "", cls: str = "th") -> str:
    if not data_uri:
        return f'<div class="{cls} missing" title="{esc(alt)}">이미지 없음</div>'
    return f'<img class="{cls}" src="{data_uri}" alt="{esc(alt)}" title="{esc(alt)}">'


def reviewed_box(item: str, label: str = "검토") -> str:
    return f'<label class="rev"><input type="checkbox" data-item="{esc(item)}" data-field="reviewed"> {esc(label)}</label>'


# ---------------------------------------------------------------- HTML 문서
CSS = """
:root{--bg:#f6f7f9;--panel:#fff;--line:#d9dde3;--ink:#1c1f24;--muted:#5c6470;--acc:#1e66f5;--ok:#1a7f37;--warn:#b26a00;--bad:#b3261e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Malgun Gothic","Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header h1{font-size:16px;margin:0 12px 0 0}header .sp{flex:1}
button{border:1px solid var(--line);background:#fff;border-radius:6px;padding:6px 10px;cursor:pointer;font:inherit}button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
main{padding:14px 16px;max-width:1500px;margin:0 auto}
.intro{background:#fffbe6;border:1px solid #f1e3a3;border-radius:8px;padding:10px 14px;margin-bottom:14px}
.intro ol{margin:6px 0 0 18px;padding:0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:10px}
.card h2{font-size:15px;margin:0 0 6px}.card .meta{color:var(--muted);font-size:12px}
.row{display:flex;gap:10px;align-items:flex-start;border-top:1px dashed var(--line);padding:8px 0}.row:first-of-type{border-top:0}
.thumbs{display:flex;gap:4px;flex-wrap:wrap}.th{height:160px;width:auto;border:1px solid var(--line);border-radius:4px;background:#111;object-fit:contain}
.th.missing{display:flex;align-items:center;justify-content:center;color:#999;width:90px;font-size:11px}
.fields{display:flex;flex-direction:column;gap:5px;min-width:250px}.fields label{display:flex;gap:6px;align-items:center;font-size:13px}
input[type=text],input[type=number],select{font:inherit;padding:4px 6px;border:1px solid var(--line);border-radius:5px}
input[type=text].gt{width:110px;font-weight:600}input[type=number]{width:100px}
.pair{display:flex;gap:14px;align-items:center}.pair .side{text-align:center;font-size:12px;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.cell{border:1px solid var(--line);border-radius:8px;padding:6px;text-align:center;background:#fff}.cell .th{height:150px}
.radios{display:flex;gap:8px;justify-content:center;flex-wrap:wrap}.radios label{font-size:12px;display:flex;gap:3px;align-items:center}
.tag{display:inline-block;padding:1px 6px;border-radius:10px;font-size:11px;background:#eef1f5;color:var(--muted);margin-left:4px}
.tag.person{background:#e6f4ea;color:var(--ok)}.tag.reject{background:#fdecea;color:var(--bad)}.tag.object{background:#e8f0fe;color:var(--acc)}.tag.warn{background:#fff4d6;color:var(--warn)}
.done{outline:2px solid #bfe3c8}.progress{font-weight:600}.rev{font-weight:600;color:var(--ok)}
.saveerr{color:var(--bad);font-weight:600}
details summary{cursor:pointer;color:var(--muted)}
"""

JS = r"""
const META = JSON.parse(document.getElementById('meta').textContent);
const KEY = 'gtsheet:' + META.store_key + ':' + (META.manifest || 'nomanifest');
const $ = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
function fields(){ return $$('[data-item][data-field]'); }
function collect(){
  const labels = {};
  fields().forEach(el => {
    const item = el.dataset.item, f = el.dataset.field;
    labels[item] = labels[item] || {};
    if (el.type === 'radio') { if (el.checked) labels[item][f] = el.value; else if (!(f in labels[item])) labels[item][f] = null; }
    else if (el.type === 'checkbox') labels[item][f] = el.checked;
    else labels[item][f] = el.value;
  });
  return labels;
}
function apply(data){
  const labels = (data && data.labels) || {};
  if (data && data.labeler != null && $('#labeler')) $('#labeler').value = data.labeler;
  fields().forEach(el => {
    const v = labels[el.dataset.item]; if (!v || !(el.dataset.field in v)) return;
    const val = v[el.dataset.field];
    if (el.type === 'radio') el.checked = (val !== null && String(val) === el.value);
    else if (el.type === 'checkbox') el.checked = !!val;
    else el.value = val == null ? '' : val;
  });
  updateProgress();
}
function save(){
  const err = $('#saveerr');
  try { localStorage.setItem(KEY, JSON.stringify({labeler: ($('#labeler')||{}).value || '', labels: collect(), saved_at: new Date().toISOString(), manifest: META.manifest})); if (err) err.textContent = ''; }
  catch(e) { if (err) err.textContent = '자동 저장 실패 (' + e.name + ') — labels.json 을 자주 내려받으세요'; }
  updateProgress();
}
function restore(){ try { const raw = localStorage.getItem(KEY); if (raw) { const d = JSON.parse(raw); if (!d.manifest || d.manifest === META.manifest) apply(d); } } catch(e) {} }
function isDone(vals){ return !!vals.reviewed; }
function updateProgress(){
  const labels = collect(); const items = Object.keys(labels); let done = 0;
  items.forEach(id => { const ok = isDone(labels[id]); if (ok) done++; $$('[data-block="' + CSS.escape(id) + '"]').forEach(b => b.classList.toggle('done', ok)); });
  const p = $('#progress'); if (p) p.textContent = done + ' / ' + items.length + ' 검토';
}
function download(name, text){
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], {type: 'application/json'})); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
function exportJson(){
  const labels = collect(); const n = Object.keys(labels).length; const done = Object.values(labels).filter(isDone).length;
  if (done < n && !confirm('검토 표시가 ' + done + ' / ' + n + ' 입니다. 검토하지 않은 항목은 정답으로 쓰이지 않습니다. 그래도 내보낼까요?')) return;
  const data = {kind: META.kind, meta: META, labeler: ($('#labeler')||{}).value || '', exported_at: new Date().toISOString(), reviewed: done, items: n, labels: labels};
  download(META.export_name, JSON.stringify(data, null, 1));
}
function importJson(file){
  const r = new FileReader(); r.onload = () => { try {
    const d = JSON.parse(r.result);
    const m = d.meta && d.meta.manifest;
    if (d.kind && d.kind !== META.kind) { alert('다른 종류의 라벨 파일입니다: ' + d.kind); return; }
    if (m && m !== META.manifest && !confirm('이 라벨 파일은 다른 버전의 시트(manifest ' + m + ')에서 나왔습니다. 항목이 달라졌을 수 있습니다. 그래도 불러올까요?')) return;
    apply(d); save(); alert('불러왔습니다');
  } catch(e) { alert('JSON 이 아닙니다: ' + e); } }; r.readAsText(file);
}
function resetAll(){
  if (!confirm('이 시트의 입력을 모두 초기 제안값으로 되돌립니다. 계속할까요?')) return;
  try { localStorage.removeItem(KEY); } catch(e) {}
  location.reload();
}
function markReviewed(root){
  $$('input[data-field="reviewed"]', root).forEach(cb => { cb.checked = true; });
  save();
}
document.addEventListener('DOMContentLoaded', () => {
  restore();
  const touch = e => {
    const el = e.target; if (!el.matches('[data-item][data-field]')) { if (el.id === 'labeler') save(); return; }
    if (el.dataset.field !== 'reviewed') { $$('input[data-field="reviewed"][data-item="' + CSS.escape(el.dataset.item) + '"]').forEach(cb => { cb.checked = true; }); }
    save();
  };
  document.addEventListener('input', touch);
  document.addEventListener('change', touch);
  $('#export').addEventListener('click', exportJson);
  $('#import').addEventListener('change', e => { if (e.target.files[0]) importJson(e.target.files[0]); e.target.value = ''; });
  $('#reset').addEventListener('click', resetAll);
  $$('[data-mark-all]').forEach(b => b.addEventListener('click', () => { if (confirm('이 묶음의 항목을 모두 검토한 것으로 표시합니다. 정말 모두 확인했나요?')) markReviewed(b.closest('[data-group]') || document); }));
  updateProgress();
});
"""


def html_document(title: str, intro_html: str, body_html: str, *, kind: str, store_key: str, export_name: str,
                  meta: Optional[Dict[str, Any]] = None, extra_js: str = "") -> str:
    meta_all = {"kind": kind, "store_key": store_key, "export_name": export_name, "generated_at": now_iso()}
    meta_all.update(meta or {})
    meta_json = json.dumps(meta_all, ensure_ascii=False, default=_json_default).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>{esc(title)}</title><style>{CSS}</style></head>
<body>
<header>
  <h1>{esc(title)}</h1>
  <label>라벨러 <input type="text" id="labeler" placeholder="이름/이니셜" style="width:110px"></label>
  <span class="progress" id="progress"></span>
  <span class="saveerr" id="saveerr"></span>
  <span class="sp"></span>
  <span class="meta">manifest {esc(meta_all.get('manifest', '-'))}</span>
  <button class="primary" id="export">labels.json 내려받기</button>
  <label><button type="button" onclick="document.getElementById('import').click()">불러오기</button><input type="file" id="import" accept=".json" style="display:none"></label>
  <button id="reset">초기화</button>
</header>
<main>
<div class="intro">{intro_html}</div>
{body_html}
</main>
<script id="meta" type="application/json">{meta_json}</script>
<script>{extra_js}</script>
<script>{JS}</script>
</body></html>
"""


def read_labels(path: Any) -> Dict[str, Dict[str, Any]]:
    """labels.json → {item_id: {field: value}}. 없으면 {}. (meta 는 read_labels_meta)"""
    labels, _meta = read_labels_meta(path)
    return labels


def read_labels_meta(path: Any) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, dict):
        return {}, {}
    labels = data.get("labels")
    meta = dict(data.get("meta") or {})
    meta.setdefault("kind", data.get("kind"))
    meta["labeler"] = data.get("labeler")
    meta["exported_at"] = data.get("exported_at")
    return {str(k): dict(v) for k, v in (labels or {}).items() if isinstance(v, dict)}, meta


LEGACY_VALID = {"verdict": {"same", "different", "unsure"}, "relevant": {"yes", "no", "unsure"}}


def is_reviewed(label: Optional[Dict[str, Any]]) -> bool:
    """항목 라벨이 사람 검토를 거쳤는가. reviewed 는 bool True 만 인정(문자열 "true"/"false" 는 검토 아님).
    reviewed 키가 아예 없는 옛 파일은 판정 필드(verdict/relevant)에 유효한 값이 있을 때만 검토로 본다."""
    if not isinstance(label, dict):
        return False
    if "reviewed" in label:
        return label.get("reviewed") is True
    return any(str(label.get(f) or "").strip().lower() in ok for f, ok in LEGACY_VALID.items())


def check_manifest(meta: Dict[str, Any], expected: Optional[str], what: str, ignore: bool = False, kind: Optional[str] = None) -> None:
    """정식 평가 전 labels.json 의 출처 검사: kind 일치, manifest 존재·일치. 어긋나면 중단(--ignore-manifest 로 경고만)."""
    got = meta.get("manifest")
    problems = []
    if kind and meta.get("kind") and meta.get("kind") != kind:
        problems.append(f"kind {meta.get('kind')} ≠ {kind} (다른 종류의 라벨 파일)")
    if expected and not got:
        problems.append("labels.json 에 manifest 가 없습니다 (옛 시트에서 내보낸 파일 — 현재 시트에서 다시 내보내세요)")
    elif expected and got != expected:
        problems.append(f"manifest {got} ≠ 현재 시트 {expected} — 시트를 다시 만든 뒤의 라벨이 아닙니다 (항목이 달라졌을 수 있음)")
    if problems:
        msg = f"{what}: " + "; ".join(problems)
        if not ignore:
            raise SystemExit(msg + ". --ignore-manifest 로 강행할 수 있습니다.")
        print("[warn] " + msg)


def fmt_time(sec: Any) -> str:
    try:
        s = float(sec)
    except (TypeError, ValueError):
        return "-"
    m, s2 = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s2:02d}" if h else f"{m:d}:{s2:02d}"
