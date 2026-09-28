"""영상 검색 결과의 영상당 표시 상한(per_video_max), 영상 이름 정규화, 확장자 없는 영상 경로 해석,
그리고 GUI ResultsPanel 의 'seek 1회' 재생 로직을 검증한다.

Qt 는 offscreen 플랫폼으로 띄운다 (실제 창 없음).
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search import unified_search_4mode as us  # noqa: E402


def _cand(video: str, gid: str) -> us.GroupCandidate:
    return us.GroupCandidate(group_id=gid, payload={"video": video, "track_key": gid})


class VideoKeyTests(unittest.TestCase):
    def test_strips_video_extension_and_directories(self):
        self.assertEqual(us.video_key_of({"video": "Normal_Videos_935_x264.mp4"}), "Normal_Videos_935_x264")
        self.assertEqual(us.video_key_of({"video": "Normal_Videos_935_x264"}), "Normal_Videos_935_x264")
        self.assertEqual(
            us.video_key_of({"video_path": r"C:\x\data\videos\Normal_Videos_935_x264.mp4"}),
            "Normal_Videos_935_x264",
        )
        self.assertEqual(us.video_key_of({"video_name": "a/b/clip.MKV"}), "clip")

    def test_non_video_suffix_is_kept(self):
        # 'x264' 같은 점 없는 이름이나 영상 확장자가 아닌 접미사는 그대로 둔다.
        self.assertEqual(us.video_key_of({"video": "cam01.v2"}), "cam01.v2")

    def test_empty_when_unknown(self):
        self.assertEqual(us.video_key_of({}), "")
        self.assertEqual(us.video_key_of(None), "")
        self.assertEqual(us.video_key_of({"video": "", "video_path": None}), "")

    def test_group_video_key_uses_first_hit(self):
        group = SimpleNamespace(id="v/person_1", hits=[SimpleNamespace(payload={"video": "v.mp4"}, score=0.9)])
        self.assertEqual(us.group_video_key(group), "v")
        self.assertEqual(us.group_video_key(SimpleNamespace(id="x", hits=[])), "")


class PerVideoCapTests(unittest.TestCase):
    def setUp(self):
        # 순위 순서: A1 A2 B1 A3 A4 C1 B2 A5 D1
        self.items = [
            _cand("A.mp4", "A/1"), _cand("A.mp4", "A/2"), _cand("B.mp4", "B/1"),
            _cand("A", "A/3"), _cand("A.mp4", "A/4"), _cand("C.mp4", "C/1"),
            _cand("B.mp4", "B/2"), _cand("A.mp4", "A/5"), _cand("D.mp4", "D/1"),
        ]
        self.key = lambda c: us.video_key_of(c.payload)

    def ids(self, rows):
        return [r.group_id for r in rows]

    def test_no_cap_is_plain_top_k(self):
        kept, hidden, total = us.select_with_per_video_cap(self.items, top_k=4, per_video_max=0, video_of=self.key)
        self.assertEqual(self.ids(kept), ["A/1", "A/2", "B/1", "A/3"])
        self.assertEqual(hidden, {})
        self.assertEqual(total, {"A": 5, "B": 2, "C": 1, "D": 1})

    def test_cap_keeps_rank_order_and_counts_hidden(self):
        kept, hidden, total = us.select_with_per_video_cap(self.items, top_k=5, per_video_max=2, video_of=self.key)
        # A 는 2개까지: A1 A2 B1 (A3 A4 접힘) C1 B2 -> 5개 채움
        self.assertEqual(self.ids(kept), ["A/1", "A/2", "B/1", "C/1", "B/2"])
        self.assertEqual(hidden, {"A": 2})
        # 'A' (확장자 없음) 과 'A.mp4' 가 같은 영상으로 합산된다
        self.assertEqual(total["A"], 5)

    def test_cap_one_gives_one_per_video(self):
        kept, hidden, _ = us.select_with_per_video_cap(self.items, top_k=10, per_video_max=1, video_of=self.key)
        self.assertEqual(self.ids(kept), ["A/1", "B/1", "C/1", "D/1"])
        self.assertEqual(hidden, {"A": 4, "B": 1})

    def test_stops_counting_hidden_once_top_k_is_full(self):
        kept, hidden, _ = us.select_with_per_video_cap(self.items, top_k=2, per_video_max=1, video_of=self.key)
        self.assertEqual(self.ids(kept), ["A/1", "B/1"])
        # B1 로 top_k 가 찼으므로 그 뒤 A3.. 는 접힘으로 세지 않는다 (A2 만 접힘)
        self.assertEqual(hidden, {"A": 1})

    def test_empty_and_zero_top_k(self):
        self.assertEqual(us.select_with_per_video_cap([], top_k=5, per_video_max=2, video_of=self.key)[0], [])
        self.assertEqual(us.select_with_per_video_cap(self.items, top_k=0, per_video_max=2, video_of=self.key)[0], [])

    def test_annotate_row(self):
        row = {"rank": 1}
        us.annotate_video_cap(row, {"video": "A.mp4"}, {"A": 2}, {"A": 5})
        self.assertEqual(row["video_key"], "A")
        self.assertEqual(row["same_video_hidden"], 2)
        self.assertEqual(row["same_video_candidates"], 5)
        row2 = us.annotate_video_cap({}, {"video": "Z.mp4"}, {}, {})
        self.assertEqual((row2["same_video_hidden"], row2["same_video_candidates"]), (0, 0))


class CliTests(unittest.TestCase):
    def test_per_video_max_default_and_validation(self):
        parser = us.build_parser()
        args = parser.parse_args(["text-video", "--scope", "person", "--text", "black bag"])
        self.assertEqual(args.per_video_max, 0)
        args = parser.parse_args(["image-video", "--scope", "person", "--image", "q.jpg", "--per-video-max", "3"])
        self.assertEqual(args.per_video_max, 3)
        bad = parser.parse_args(["text-video", "--scope", "person", "--text", "x", "--per-video-max", "-1"])
        with self.assertRaises(SystemExit):
            us.validate_args(bad, parser)


class ResolverTests(unittest.TestCase):
    def test_resolve_video_without_extension(self):
        import search_gui

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "videos").mkdir()
            target = root / "videos" / "Normal_Videos_935_x264.mp4"
            target.write_bytes(b"")
            resolver = search_gui.PathResolver(root)
            resolver.set_video_root(root / "videos")
            # final_db_candidates 계열 payload: video 에 확장자 없음, video_path 빈 문자열
            row = {"video": "Normal_Videos_935_x264", "video_path": "",
                   "payload": {"video": "Normal_Videos_935_x264", "video_path": ""}}
            self.assertEqual(resolver.resolve_video(row), target)
            # 확장자가 있는 기존 payload 도 그대로 동작
            row2 = {"payload": {"video": "Normal_Videos_935_x264.mp4", "video_path": r"D:\old\videos\Normal_Videos_935_x264.mp4"}}
            self.assertEqual(resolver.resolve_video(row2), target)
            # 없는 영상은 None
            self.assertIsNone(resolver.resolve_video({"payload": {"video": "nope_000"}}))


class _FakePlayer:
    def __init__(self, status):
        self.status = status
        self.calls = []

    def setSource(self, url):  # noqa: N802
        self.calls.append(("setSource", url.toLocalFile()))

    def setPosition(self, ms):  # noqa: N802
        self.calls.append(("setPosition", int(ms)))

    def play(self):
        self.calls.append(("play",))

    def pause(self):
        self.calls.append(("pause",))

    def stop(self):
        self.calls.append(("stop",))

    def mediaStatus(self):  # noqa: N802
        return self.status

    def count(self, name):
        return sum(1 for c in self.calls if c[0] == name)


@unittest.skipIf(os.environ.get("SKIP_QT_TESTS") == "1", "Qt tests disabled")
class SeekOnceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        import search_gui

        cls.gui = search_gui
        from PySide6.QtMultimedia import QMediaPlayer

        cls.MS = QMediaPlayer.MediaStatus

    def _panel(self, status):
        panel = self.gui.ResultsPanel(video=True)
        if panel.player is not None:
            panel.player.stop()
        fake = _FakePlayer(status)
        panel.player = fake
        return panel, fake

    def test_status_storm_seeks_only_once(self):
        panel, fake = self._panel(self.MS.LoadingMedia)
        path = Path(tempfile.gettempdir()) / "clip_a.mp4"
        panel._load_video(path, 449.0, autoplay=False)
        self.assertEqual(fake.count("setSource"), 1)
        self.assertEqual(fake.count("setPosition"), 0)  # 로딩 중에는 seek 하지 않음
        self.assertTrue(panel._seek_pending)

        fake.status = self.MS.LoadedMedia
        panel._on_media_status_changed(self.MS.LoadedMedia)
        self.assertEqual(fake.calls[-2:], [("setPosition", 449000), ("pause",)])
        self.assertFalse(panel._seek_pending)

        # FFmpeg 백엔드가 보내는 상태 반복: 더 이상 seek/pause 하지 않아야 한다
        for st in (self.MS.BufferingMedia, self.MS.BufferedMedia, self.MS.LoadedMedia,
                   self.MS.BufferingMedia, self.MS.BufferedMedia):
            fake.status = st
            panel._on_media_status_changed(st)
        self.assertEqual(fake.count("setPosition"), 1)
        self.assertEqual(fake.count("pause"), 1)

    def test_play_button_on_loaded_video_seeks_and_plays_immediately(self):
        panel, fake = self._panel(self.MS.LoadingMedia)
        path = Path(tempfile.gettempdir()) / "clip_b.mp4"
        panel._load_video(path, 10.0, autoplay=False)
        fake.status = self.MS.BufferedMedia
        panel._on_media_status_changed(self.MS.LoadedMedia)
        self.assertEqual(fake.count("pause"), 1)

        # '해당 시점 재생' 클릭에 해당: 같은 파일이 준비돼 있으면 즉시 seek + play
        panel._load_video(path, 10.0, autoplay=True)
        self.assertEqual(fake.count("setSource"), 1)
        self.assertEqual(fake.calls[-2:], [("setPosition", 10000), ("play",)])
        self.assertFalse(panel._seek_pending)

        # 재생 중 상태 반복은 위치를 되돌리지 않는다
        for st in (self.MS.LoadedMedia, self.MS.BufferingMedia, self.MS.BufferedMedia):
            panel._on_media_status_changed(st)
        self.assertEqual(fake.count("setPosition"), 2)
        self.assertEqual(fake.count("play"), 1)
        self.assertEqual(fake.count("pause"), 1)

    def test_play_while_still_loading_defers_until_loaded(self):
        panel, fake = self._panel(self.MS.LoadingMedia)
        path = Path(tempfile.gettempdir()) / "clip_c.mp4"
        panel._load_video(path, 5.0, autoplay=False)
        panel._load_video(path, 5.0, autoplay=True)  # 아직 LoadingMedia
        self.assertEqual(fake.count("setPosition"), 0)
        self.assertTrue(panel._seek_pending)
        fake.status = self.MS.LoadedMedia
        panel._on_media_status_changed(self.MS.LoadedMedia)
        self.assertEqual(fake.calls[-2:], [("setPosition", 5000), ("play",)])

    def test_switching_result_reseeks_once(self):
        panel, fake = self._panel(self.MS.BufferedMedia)
        path = Path(tempfile.gettempdir()) / "clip_d.mp4"
        panel._loaded_video = path.resolve()
        panel._load_video(path, 1.0, autoplay=False)
        panel._load_video(path, 2.0, autoplay=False)
        self.assertEqual([c for c in fake.calls if c[0] == "setPosition"], [("setPosition", 1000), ("setPosition", 2000)])
        self.assertEqual(fake.count("setSource"), 0)

    def test_invalid_media_reports_in_detail_and_clears_pending(self):
        panel, fake = self._panel(self.MS.LoadingMedia)
        panel._load_video(Path(tempfile.gettempdir()) / "broken.mp4", 3.0, autoplay=True)
        panel._on_media_status_changed(self.MS.InvalidMedia)
        self.assertFalse(panel._seek_pending)
        self.assertIn("재생 오류", panel.detail.toPlainText())
        self.assertEqual(fake.count("play"), 0)

    def test_clear_resets_pending(self):
        panel, fake = self._panel(self.MS.LoadingMedia)
        panel._load_video(Path(tempfile.gettempdir()) / "clip_e.mp4", 3.0, autoplay=True)
        panel.clear()
        self.assertFalse(panel._seek_pending)
        self.assertIsNone(panel._loaded_video)


if __name__ == "__main__":
    unittest.main()
