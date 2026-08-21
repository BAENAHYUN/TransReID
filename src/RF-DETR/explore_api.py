"""
1단계: rfdetr 패키지에 decoder layer/query 조절 기능이 실제로 있는지 확인
"""
import inspect
from rfdetr import RFDETRMedium

model = RFDETRMedium()

print("=== model이 가진 메서드/속성 전체 목록 ===")
for m in dir(model):
    if not m.startswith("_"):
        print(m)

print("\n=== predict() 함수가 받는 인자들 ===")
print(inspect.signature(model.predict))

print("\n=== inference() 함수가 받는 인자들 ===")
print(inspect.signature(model.inference))


print("\n=== model_config 내용 ===")
print(model.model_config)

print("\n=== model_config 안의 속성들 ===")
for attr in dir(model.model_config):
    if not attr.startswith("_"):
        print(attr, "=", getattr(model.model_config, attr, "?"))