"""오프라인 테스트: video/sushi_inference.link_windows — 512 프레임 창 경계에서 raw track 연속성으로 long id 를 잇는 후처리 (SUSHI·torch 불필요한 순수 함수)."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module():
    """torch 가 없거나 무거워도 함수만 검사할 수 있게 stub 으로 import."""
    for name in ("torch",):
        if name not in sys.modules:
            try:
                __import__(name)
            except Exception:
                sys.modules[name] = types.ModuleType(name)
    spec = importlib.util.spec_from_file_location("sushi_inference_under_test", ROOT / "video" / "sushi_inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = load_module()


def det(did, frame, tid, x, y=0.0, w=10.0, h=20.0):
    return {"detection_id": did, "frame": frame, "source_track_id": tid, "bb_left": x, "bb_top": y, "bb_right": x + w, "bb_bot": y + h}


def window(rows):
    df = pd.DataFrame(rows)
    return (int(df.frame.min()), int(df.frame.max()), df)


class LinkTests(unittest.TestCase):
    def test_box_iou(self):
        self.assertAlmostEqual(M.box_iou((0, 0, 10, 20), (0, 0, 10, 20)), 1.0)
        self.assertAlmostEqual(M.box_iou((0, 0, 10, 20), (5, 0, 15, 20)), 1 / 3, places=4)
        self.assertEqual(M.box_iou((0, 0, 10, 20), (50, 0, 60, 20)), 0.0)

    def test_links_same_raw_track_across_boundary(self):
        # 창1: raw 0 (사람 A) frames 0..511 long 1 · raw 1 (사람 B) long 2 / 창2: raw 0 이어짐 long 3 · raw 7 새 사람 long 4
        w1 = window([det(1, 510, 0, 100.0), det(2, 511, 0, 100.0), det(3, 511, 1, 300.0)])
        w2 = window([det(4, 512, 0, 100.0), det(5, 513, 0, 100.0), det(6, 512, 7, 500.0)])
        d2l = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4}
        remap, st = M.link_windows([w1, w2], d2l, max_gap=2, min_iou=0.3)
        self.assertEqual(st["linked"], 1)
        self.assertEqual(st["candidates"], 1)
        self.assertEqual(remap[1], remap[3])
        self.assertNotEqual(remap[1], remap[2])
        self.assertEqual(sorted(set(remap.values())), [1, 2, 3])            # 4 -> 3 개로 압축, 1 부터
        self.assertEqual((st["long_ids_before"], st["long_ids_after"]), (4, 3))

    def test_rejects_gap_and_iou(self):
        w1 = window([det(1, 500, 0, 100.0)])                                 # 마지막 검출이 경계에서 12 프레임 전
        w2 = window([det(2, 512, 0, 100.0)])
        remap, st = M.link_windows([w1, w2], {1: 1, 2: 2}, max_gap=2, min_iou=0.3)
        self.assertEqual((st["linked"], st["rejected_gap"]), (0, 1))
        self.assertNotEqual(remap[1], remap[2])
        w1 = window([det(1, 511, 0, 100.0)])
        w2 = window([det(2, 512, 0, 400.0)])                                 # 같은 raw id 지만 박스가 멀다
        remap, st = M.link_windows([w1, w2], {1: 1, 2: 2}, max_gap=2, min_iou=0.3)
        self.assertEqual((st["linked"], st["rejected_iou"]), (0, 1))

    def test_one_to_one_conflict(self):
        # 창1 의 long 1 이 창2 에서 두 long id(3, 4) 로 갈라진 두 raw track 과 이어지려 함 → IoU 큰 쪽만 잇고 나머지는 conflict
        w1 = window([det(1, 511, 0, 100.0), det(2, 511, 1, 104.0)])
        w2 = window([det(3, 512, 0, 100.0), det(4, 512, 1, 104.0)])
        d2l = {1: 1, 2: 1, 3: 3, 4: 4}
        remap, st = M.link_windows([w1, w2], d2l, max_gap=2, min_iou=0.3)
        self.assertEqual(st["linked"], 1)
        self.assertEqual(st["conflicts"], 1)
        self.assertEqual(len({remap[1], remap[3], remap[4]}), 2)

    def test_chain_over_three_windows(self):
        w1 = window([det(1, 511, 0, 100.0)])
        w2 = window([det(2, 512, 0, 100.0), det(3, 1023, 0, 100.0)])
        w3 = window([det(4, 1024, 0, 100.0)])
        remap, st = M.link_windows([w1, w2, w3], {1: 1, 2: 2, 3: 2, 4: 3}, max_gap=2, min_iou=0.3)
        self.assertEqual(st["linked"], 2)
        self.assertEqual(len(set(remap.values())), 1)

    def test_single_window_noop(self):
        w1 = window([det(1, 0, 0, 100.0)])
        remap, st = M.link_windows([w1], {1: 5}, max_gap=2, min_iou=0.3)
        self.assertEqual(st["boundaries"], 0)
        self.assertEqual(remap, {5: 1})

    def test_preprocess_yaml_flag(self):
        from video import batch_preprocess_videos_parallel as B
        self.assertEqual(B.stitcher_link_args(ROOT / "pipeline_tracking.yaml"), [])
        args = B.stitcher_link_args(ROOT / "pipeline_tracking_sushi_link.yaml")
        self.assertEqual(args[:1], ["--link-windows"])
        self.assertIn("--link-max-gap", args)
        self.assertEqual(B.stitcher_link_args(ROOT / "no_such.yaml"), [])


if __name__ == "__main__":
    unittest.main()
