# 🧬 양자 데이터 인코딩(Embedding) 방법

## 1️⃣ Amplitude Encoding (진폭 인코딩)

**💡 설명**

- 고전 데이터 벡터를 양자 상태의 진폭(amplitude)에 직접 매핑합니다.
- 예를 들어, 데이터 벡터 $[x_0, x_1, …, x_{N-1}]$를 정규화하여 $\sum_{i=0}^{N-1} x_i^2 = 1$를 만족시키고, 이를 양자 상태 $|x\rangle = \sum_{i=0}^{N-1} x_i |i\rangle$로 변환.

**✅ 장점**

- 매우 높은 데이터 압축률: N개의 고전 데이터를 \log_2 N 큐빗으로 표현 가능 (예: 1024차원 → 10 큐빗)
- 복잡한 고차원 정보를 한 번에 인코딩 가능

**⚠️ 단점**

- 회로 설계가 복잡하며, 실제 물리적 구현 어려움
- 진폭 정규화 필요 → 데이터 준비에 추가 전처리 필요
- 실제 양자 하드웨어에서 구현 시 에러에 매우 민감

## 2️⃣ Angle Encoding (Angle or Rotation Encoding)

**💡 설명**

- 고전 데이터 값 x_i를 단일 큐빗의 rotation gate 각도로 매핑 (예: R_y(x_i), R_z(x_i))
- 예를 들어, x_i를 R_y(x_i) gate의 rotation parameter로 사용

**✅ 장점**

- 간단한 회로 구조, 하드웨어에 잘 맞음
- 직관적인 해석 (데이터 → 회전 각도)

**⚠️ 단점**
- 큐빗 수가 데이터 feature 수만큼 필요, feature 수가 많으면 큐빗이 급격히 증가
- 데이터 크기가 크면 차원 축소 필요 (예: PCA)

## 3️⃣ Basis Encoding (Computational Basis Encoding)

**💡 설명**

- 데이터를 이진 표현으로 변환 후 큐빗의 computational basis state에 직접 할당
- 예: 데이터가 3이면 $|11\rangle$에 해당

**✅ 장점**

- 가장 단순하고 직관적

**⚠️ 단점**

- 연속형 feature 데이터 표현에 부적합
- 고차원 데이터에는 큐빗 수가 매우 많이 필요

## 4️⃣ QSample Encoding

**💡 설명**

- 데이터 벡터를 확률 분포로 해석하고, 샘플링 기반으로 양자 상태에 반영
- 확률분포를 샘플링하여 basis state에 투영

**✅ 장점**

- 확률적 데이터 표현에 유용
- 샘플링 기반으로 noise-resilient

**⚠️ 단점**

- 샘플링 수가 많으면 비용 증가
- 정확도 확보가 어렵고, variance 큼

## 5️⃣ IQP-style Embedding (Instantaneous Quantum Polynomial-time)

**💡 설명**

- 특수한 고정 구조의 양자 회로에 classical data를 phase shift 형태로 삽입
- 특히 Fourier-type embedding으로 쓰임

**✅ 장점**

- 복잡한 다변수 의존성 표현 가능
- 데이터의 비선형 효과를 자연스럽게 반영

**⚠️ 단점**

- 회로 깊이가 커지고, 해석이 복잡
- 일반적으로 적용하기 위해 hyperparameter 튜닝 필요

## 6️⃣ Hamiltonian Encoding

**💡 설명**

- 데이터에 따라 특정 Hamiltonian을 설계, 해당 Hamiltonian의 time-evolution operator e^{-iHt}로 인코딩

**✅ 장점**

- 물리 기반 해석 가능
- 동역학 시스템 표현에 유리

**⚠️ 단점**

- 설계가 복잡
- 고전 데이터에 대한 직관적 해석 어려움

## ✨ 정리 요약 표

|인코딩 종류|설명|장점|단점|
|:---:|:---:|:---:|:---:|
|Amplitude Encoding|진폭에 직접 매핑|고차원 데이터 압축, 큐빗 절약|회로 복잡, 정규화 필요|
|Angle Encoding|rotation gate 각도에 매핑|간단, 직관적|큐빗 수 많이 필요, 차원 축소 필요|
|Basis Encoding|이진 표현 후 basis state 할당|단순, 직관적|연속 데이터 부적합, 큐빗 폭증|
|QSample Encoding|확률분포로 샘플링 후 투영|확률 기반, noise-resilient|비용 증가, variance 큼|
|IQP-style Embedding|phase shift 기반 임베딩|비선형성 표현, 고급 표현력|깊이 증가, 해석 복잡|
|Hamiltonian Encoding|Hamiltonian time evolution 사용|물리 해석 가능|설계 복잡, 직관적 해석 어려움|
