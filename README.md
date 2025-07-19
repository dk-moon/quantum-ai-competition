# Quantum AI Competition

## 1. 목적 및 범위

- Fashion-MNIST 0 (T-shirt/top) vs 6 (Shirt) 이진 분류 문제를 양자-고전 결합 모델로 해결하는 과정 전체를 제시합니다.
- 코드는 PennyLane과 PyTorch를 연동하여, 양자 회로(QNN) 를 고전 CNN 뒤에 연결하는 전형적 구조를 보여 줍니다.
- 주요 목표는 “대회 규격 준수”와 “엔드-투-엔드 파이프라인 제공”입니다. 즉, 참가자께서는 본 예제를 기반으로 자신만의 회로·전처리를 교체해 성능을 향상시키실 수 있습니다.

## 2. 환경 설정

|단계|설명|
|---|---|
|라이브러리 설치|pip install pennylane torch torchvision<br>별도 옵션 없이 실행하면 CPU 환경에서도 문제없이 작동합니다.|
|디바이스 선언|python_dev = qml.device("default.qubit", wires=3)<br>- default.qubit: 잡음 없는 범용 시뮬레이터<br>- wires=3: 3 큐빗 사용(예시)|

## 3. 양자 회로 구성 개념
@qml.qnode 데코레이터

파이썬 함수를 “양자 회로”로 선언합니다.
게이트 배치

예) qml.H, qml.RX, qml.CNOT 등으로 회로 동작을 정의합니다.
측정 방법

qml.probs() : 전체 상태의 확률 분포
qml.expval(qml.PauliZ(0)) : 특정 관측치 기대값
> 핵심: 회로는 “레이어”처럼 동작하므로, PyTorch와 자연스럽게 이어집니다.

## 4. 데이터 전처리
|절차|세부 내용|
|---|---|
|다운로드|torchvision.datasets.FashionMNIST(train/test)|
|필터링|라벨 0·6만 선택 → 이진 문제로 단순화|
|라벨 변환|라벨 6을 1로 매핑(0 ↔ 1)|
|배치 구성|DataLoader로 학습·추론 배치 제공|

## 5. 모델 아키텍처 개요
- 고전 CNN 전처리
    - 두 차례 Convolution → MaxPooling → Flatten
    - 특징 벡터를 2차원으로 축소
- 양자 회로(QNN)
    - 2 큐빗, 파라미터 8개
    - 입력 벡터를 게이트 회전에 매핑 → PauliZ⊗PauliZ 기대값 측정
- 출력 변환
    - 선형 계층 1 개로 0/1 확률 구성 → log_softmax 반환

## 6. 대회 규격 자동 검증
```
specs = qml.specs(bc.qnn)(dummy_x)
assert specs["num_tape_wires"] <= 8,  "❌ 큐빗 수 초과"
assert specs['resources'].depth <= 30, "❌ 회로 깊이 초과"
assert specs["num_trainable_params"]<= 60, "❌ 학습 퀀텀 파라미터 수 초과"
assert total_params <= 50000, "❌ 학습 전체 파라미터 수 초과" 추론 및 제출 파일 생성
```

- 전체 테스트 세트 추론 후 0·6 라벨만 평가
- 1→6 라벨 복원 → 정확도 출력
- 파일명: y_pred_YYYYMMDD_HHMMSS.csv

## 8. 확장 안내(개념 위주)
- 다양한 인코딩 전략: Amplitude Encoding 등으로 픽셀 정보를 직접 큐빗 상태에 매핑 가능
- 회로 최적화: 불필요한 CNOT 줄이기, 파라미터 공유로 규격 내 성능 극대화
- 하이퍼파라미터: 학습률·드롭아웃 비율 등 조정 → 과적합 방지

## 9. 요약
- 본 베이스라인은 “학습 → 검증 → 추론 → 제출” 전 과정을 담은 참조 구현입니다.
- 구조를 이해하신 뒤, CNN 블록 또는 QNN 회로를 자유롭게 변형하여 성능을 개선하시기 바랍니다.
- 모든 변경 시 큐빗 수・회로 깊이・파라미터 수가 규격 내에 있는지 반드시 확인하세요.