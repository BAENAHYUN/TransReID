"""report/build_image_results.py (결과창 한 번에) — dry-run 계획: DB 리포트 → assignments 있는 갤러리만 → 인덱스, --reuse 건너뜀, 마커 접두어."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from report import build_image_results as R  # noqa: E402


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        (self.root / "cl" / "person").mkdir(parents=True)
        (self.root / "cl" / "person" / "person_leiden_assignments.jsonl").write_text("{}\n", encoding="utf-8")

    def tearDown(self):
        self.td.cleanup()

    def run_dry(self, *extra):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = R.main(["--dry-run", "--cluster-dir", str(self.root / "cl"), "--out-dir", str(self.root / "out"), "--source", "prw_image", *extra])
        return code, buf.getvalue()

    def test_plan_order_and_skips(self):
        code, out = self.run_dry()
        self.assertEqual(code, 0)
        lines = [l for l in out.splitlines() if l.startswith("[결과창]")]
        self.assertTrue(lines[0].startswith("[결과창] DB 리포트:") and "build_image_db_html.py" in lines[0] and "--source prw_image" in lines[0])
        self.assertTrue(lines[1].startswith("[결과창] person 갤러리:") and "--target person" in lines[1] and "--inline-images" in lines[1])
        self.assertTrue(lines[2].startswith("[결과창] object 갤러리: assignments 없음"))
        self.assertTrue(lines[3].startswith("[결과창] 결과 인덱스:") and "build_image_review_index.py" in lines[3])
        self.assertIn("image_db_prw_image.html", lines[0])
        self.assertIn("index_prw_image.html", lines[3])
        self.assertNotIn("RESULT_HTML:", out)                       # dry-run 은 마커를 찍지 않는다

    def test_reuse_skips_existing(self):
        (self.root / "out").mkdir()
        (self.root / "out" / "image_db_prw_image.html").write_text("x", encoding="utf-8")
        code, out = self.run_dry("--reuse")
        self.assertEqual(code, 0)
        self.assertIn("[결과창] DB 리포트: 이미 있음", out)
        self.assertIn("[결과창] person 갤러리:", out)                 # 갤러리 HTML 은 없으므로 실행 계획

    def test_marker_lines_are_prefixed(self):
        self.assertTrue(R.MARKER_RE.match("RESULT_HTML: C:/x.html"))
        self.assertTrue(R.MARKER_RE.match("  RESULT_SUMMARY: y.json"))
        self.assertFalse(R.MARKER_RE.match("done RESULT_HTML"))


if __name__ == "__main__":
    unittest.main()
