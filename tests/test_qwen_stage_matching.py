"""qwen_stage 의 순수 로직 검증 (모델 로드 없음).

- objects_match / find_check: 쿼리 해석('backpack') 과 관찰('bag') 의 객체 이름이 달라도 같은 attribute 로 매칭
- score_observation: 위 매칭이 반영돼 PASS 가 나오고, 형제 객체(backpack vs handbag)는 매칭하지 않음
- run() 의 Reranker 인자 기본값 (qwen_crop_stage 호환)
- _is_memory_error: 메모리 부족 예외 판별
"""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifiers import qwen_stage as q  # noqa: E402


def _chk(obj, attr, observed, visibility="sufficient"):
    return {"object": obj, "attribute": attr, "observed": observed, "visibility": visibility, "evidence": "x"}


class ObjectsMatchTests(unittest.TestCase):
    def test_hypernym_both_directions(self):
        self.assertTrue(q.objects_match("backpack", "bag"))
        self.assertTrue(q.objects_match("bag", "backpack"))
        self.assertTrue(q.objects_match("shoulder bag", "bag"))
        self.assertTrue(q.objects_match("sneakers", "shoes"))

    def test_synonyms_and_exact(self):
        self.assertTrue(q.objects_match("rucksack", "backpack"))
        self.assertTrue(q.objects_match("T-shirt", "tshirt"))
        self.assertTrue(q.objects_match("Bag", "bag"))

    def test_siblings_and_unrelated_do_not_match(self):
        self.assertFalse(q.objects_match("backpack", "handbag"))
        self.assertFalse(q.objects_match("shirt", "pants"))
        self.assertFalse(q.objects_match("", "bag"))
        self.assertFalse(q.objects_match(None, "bag"))


class FindCheckTests(unittest.TestCase):
    def test_exact_key_preferred(self):
        c = q.Constraint(object="shirt", attribute="color", expected="blue")
        checks = [_chk("top", "color", "red"), _chk("shirt", "color", "blue")]
        self.assertEqual(q.find_check(c, checks)["observed"], "blue")

    def test_falls_back_to_object_match_same_attribute(self):
        c = q.Constraint(object="backpack", attribute="color", expected="black")
        checks = [_chk("bag", "color", "black"), _chk("bag", "present", True)]
        self.assertEqual(q.find_check(c, checks)["observed"], "black")

    def test_no_match_returns_none(self):
        c = q.Constraint(object="backpack", attribute="color", expected="black")
        self.assertIsNone(q.find_check(c, [_chk("handbag", "color", "black"), _chk("shirt", "color", "black")]))
        self.assertIsNone(q.find_check(c, [{"object": "", "attribute": "color"}, "junk"]))


class ScoreObservationTests(unittest.TestCase):
    def test_renamed_object_scores_pass(self):
        cons = [q.Constraint(object="backpack", attribute="color", expected="black", required=True)]
        obs = {"inventory": ["shirt", "pants", "bag"], "checks": [_chk("bag", "color", "black")]}
        res = q.score_observation(cons, obs)
        self.assertEqual(res["attr_score"], 1.0)
        self.assertEqual(res["details"][0]["verdict"], q.PASS)
        self.assertEqual(res["details"][0]["observed_object"], "bag")
        self.assertEqual(res["failed_required"], [])

    def test_sibling_object_stays_unknown(self):
        cons = [q.Constraint(object="backpack", attribute="color", expected="black", required=True)]
        obs = {"inventory": ["handbag"], "checks": [_chk("handbag", "color", "black")]}
        res = q.score_observation(cons, obs)
        self.assertIsNone(res["attr_score"])
        self.assertEqual(res["details"][0]["verdict"], q.UNKNOWN)

    def test_presence_inventory_crosscheck_uses_object_match(self):
        cons = [q.Constraint(object="backpack", attribute="present", expected=True)]
        # inventory 에 'bag' 만 있어도 backpack present=true 를 맥락 추론으로 강등하지 않는다
        res = q.score_observation(cons, {"inventory": ["shirt", "bag"], "checks": [_chk("bag", "present", True)]})
        self.assertEqual(res["details"][0]["verdict"], q.PASS)
        # inventory 에 전혀 없으면 UNKNOWN
        res2 = q.score_observation(cons, {"inventory": ["shirt"], "checks": [_chk("backpack", "present", True)]})
        self.assertEqual(res2["details"][0]["verdict"], q.UNKNOWN)


class RunSignatureTests(unittest.TestCase):
    def test_reranker_args_have_defaults(self):
        sig = inspect.signature(q.run)
        for name in ("rerank_k", "use_reranker", "reranker_model_id", "reranker_instruction", "reranker_max_pixels"):
            self.assertIsNot(sig.parameters[name].default, inspect.Parameter.empty, name)
        self.assertTrue(sig.parameters["use_reranker"].default)
        # crop 경로가 캡션용 인스턴스를 재사용해 프로세스당 1회만 로드하도록 하는 인자
        self.assertIsNone(sig.parameters["qwen_instance"].default)


class MemoryErrorDetectTests(unittest.TestCase):
    def test_detects_cpu_and_cuda_oom(self):
        self.assertTrue(q._is_memory_error(RuntimeError("[enforce fail at alloc_cpu.cpp:117] DefaultCPUAllocator: not enough memory")))
        self.assertTrue(q._is_memory_error(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")))
        self.assertTrue(q._is_memory_error(MemoryError()))

    def test_other_errors_are_not_memory(self):
        self.assertFalse(q._is_memory_error(TypeError("unexpected keyword argument 'device_map'")))
        self.assertFalse(q._is_memory_error(ValueError("bad")))


class LoaderGuardTests(unittest.TestCase):
    def test_class_model_type_table(self):
        self.assertEqual(q.QwenVL._CLASS_MODEL_TYPE["Qwen3VLForConditionalGeneration"], "qwen3_vl")
        self.assertEqual(q.QwenVL._CLASS_MODEL_TYPE["Qwen2_5_VLForConditionalGeneration"], "qwen2_5_vl")


if __name__ == "__main__":
    unittest.main()
