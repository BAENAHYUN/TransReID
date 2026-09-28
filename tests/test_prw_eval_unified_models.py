"""오프라인 테스트: eval/prw_eval_unified.py 의 모델 목록이 yaml(person scope retriever) 기반으로 늘어나고, 새 임베더 변형(단독·후보→SOLIDER·STEP1→새 임베더·전체 RRF)이 생기는지."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench import register as R  # noqa: E402
from eval import prw_eval_unified as U  # noqa: E402


class ModelListTests(unittest.TestCase):
    def test_core_only_with_production_yaml(self):
        self.assertEqual(U.models_from_config(ROOT / "pipeline.yaml"), list(U.MODELS))
        self.assertEqual(U.models_from_config(ROOT / "pipeline.yaml", "x,siglip2"), list(U.MODELS) + ["x"])
        self.assertEqual(U.models_from_config(ROOT / "no_such.yaml", None, log=lambda *_: None), list(U.MODELS))

    def test_registered_embedder_is_included(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shutil.copyfile(ROOT / "pipeline.yaml", root / "pipeline.yaml")
            spec = {"module": "embedders.siglip2_embedder", "class": "SigLIP2Embedder", "params": {"model_id": "google/siglip2-base-patch16-224"}}
            out = R.write_embedder_yaml("myemb", spec, 768, "person", False, 1.0, root, root / "pipeline.yaml")
            self.assertEqual(U.models_from_config(out), list(U.MODELS) + ["myemb"])
            out2 = R.write_embedder_yaml("objonly", spec, 768, "object", False, 1.0, root, root / "pipeline.yaml")
            self.assertEqual(U.models_from_config(out2), list(U.MODELS))          # object scope 는 사람 검색 평가에 안 들어감


class VariantTests(unittest.TestCase):
    def test_extra_model_variants(self):
        rng = np.random.default_rng(0)
        Q, G = 6, 40
        g_pids = np.array([i % 10 for i in range(G)])
        q_pids = np.array([0, 1, 2, 3, 4, 5])
        g_frames = np.array([f"g{i}" for i in range(G)])
        q_frames = np.array([f"q{i}" for i in range(Q)])
        sims = {m: rng.random((Q, G)).astype(np.float32) for m in ("siglip2", "irra", "solider", "myemb")}
        res = U.run_variants(sims, q_pids, q_frames, g_pids, g_frames, {"siglip2": 1.0, "irra": 1.5, "solider": 1.5, "myemb": 1.0}, 2.0, 20, 20, [10],
                             models=["siglip2", "irra", "solider", "myemb"])
        for name in ("single:myemb", "myemb_prefetch_solider", "stage1_rrf_myemb_rerank", "rrf_all", "rrf_all_solider_rerank", "unified", "rrf3"):
            self.assertIn(name, res, name)
            self.assertIn("mAP", res[name])
        base = U.run_variants({m: sims[m] for m in U.MODELS}, q_pids, q_frames, g_pids, g_frames, {"siglip2": 1.0, "irra": 1.5, "solider": 1.5}, 2.0, 20, 20, [10])
        self.assertNotIn("rrf_all", base)                                          # 핵심 3종만이면 변형도 그대로
        self.assertEqual(base["unified"]["mAP"], res["unified"]["mAP"])           # 새 모델이 있어도 기존 변형 값은 같다


if __name__ == "__main__":
    unittest.main()
