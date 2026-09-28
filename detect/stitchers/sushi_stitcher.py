from __future__ import annotations

from typing import List, Sequence

from detect.base import BaseStitcher, LongTrack, Track


class SUSHIStitcher(BaseStitcher):
    """
    SUSHI 공식 코드 연결용 adapter 자리.

    다음 단계:
      1) BoT-SORT Track -> SUSHI graph/input
      2) SUSHI inference
      3) SUSHI output -> LongTrack
    """

    def __init__(
        self,
        sushi_root: str = "./third_party/SUSHI",
        checkpoint: str | None = None,
        window_sec: float = 30.0,
        **kwargs,
    ):
        self.sushi_root = sushi_root
        self.checkpoint = checkpoint
        self.window_sec = float(window_sec)
        self.kwargs = kwargs

    def stitch(self, tracks: Sequence[Track]) -> List[LongTrack]:
        raise NotImplementedError(
            "SUSHI 공식 코드 연결 전입니다. "
            "현재는 --no-stitch 로 RF-DETR + BoT-SORT smoke test를 실행하세요."
        )
