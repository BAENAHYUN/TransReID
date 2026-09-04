SCVD INGEST V2
==============

목표
----
SCVD_converted의 Train만 기존 통합 Qdrant에 append한다.

Person:
  forensic_person_embeddings_full_v1
    semantic     = SigLIP2
    person_text  = IRRA
    person_reid  = SOLIDER

Object:
  forensic_object_embeddings_full_v1
    semantic     = SigLIP2
    instance     = DINOv2

Test split은 검색/평가용으로 DB에서 제외한다.


V2에서 수정된 점
----------------
1. Train ONLY
2. Object V1 manifest 재사용
   - 기존 smoke test에서 처리한 n001~n003은 자동 skip
3. Person pass1/pass2는 Qdrant deterministic point ID를 직접 조회
   - 이미 들어간 crop vector는 skip
   - 중간에 중단되어도 재실행 가능
4. Person tracker V4의 legacy forensic_video_tracks_v4 Qdrant write 비활성화
   - SCVD에서는 track crop 생성만 수행
5. 기존 잘못 들어간 SCVD Test point를 확인/삭제하는 별도 도구 제공


파일 위치
---------
아래 파일 전부:
  C:\Users\Karsel\Desktop\Final_tool\TransReID\src\RF-DETR\


사전 확인
---------
Qdrant:
  docker start qdrant_server

컬렉션:
  forensic_person_embeddings_full_v1
  forensic_object_embeddings_full_v1


먼저 Test point가 기존 DB에 들어갔는지 확인
-----------------------------------------
python -u .\src\RF-DETR\purge_scvd_test_points_v2.py

출력이 모두 0이면 그대로 진행.

Test points가 있고 정말 평가용 Test를 비우고 싶으면:
python -u .\src\RF-DETR\purge_scvd_test_points_v2.py --yes


V2 smoke test
-------------
python -u .\src\RF-DETR\ingest_scvd_v2.py --test

주의:
--test는 "미처리 Train 영상 중 3개"를 처리한다.
object 쪽 n001~n003은 V1 manifest에 이미 완료 기록이 있으면 자동 skip될 수 있다.


전체 Train ingest
-----------------
python -u .\src\RF-DETR\ingest_scvd_v2.py


중간에 꺼졌으면
---------------
같은 명령을 다시 실행:
python -u .\src\RF-DETR\ingest_scvd_v2.py

Object:
  manifest로 완료 video skip

Person pass1/pass2:
  Qdrant에서 이미 존재하는 named vector를 직접 확인하여 skip

Person track:
  V2 manifest에 완료된 batch video는 skip.
  tracking 중 강제 종료된 batch는 다시 tracking될 수 있지만
  unified Qdrant person points는 deterministic ID이므로 중복 생성되지 않는다.


강제 재실행
-----------
python -u .\src\RF-DETR\ingest_scvd_v2.py --force

평소에는 --force 사용하지 않는 것을 권장.


DB에 넣지 않는 것
-----------------
data\scvd\SCVD_converted\Test
data\scvd\SCVD_converted_sec_split

Test는 외부 영상 검색/held-out 평가에 사용.
