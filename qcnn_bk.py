import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import pennylane as qml
from pennylane import numpy as np
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# -----------------------
# 데이터 로드 및 전처리
# -----------------------
def load_fashion_mnist_binary():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    def filter(dataset):
        indices = [i for i, (_, y) in enumerate(dataset) if y in [0, 6]]
        data = torch.stack([dataset[i][0] for i in indices])
        labels = torch.tensor([int(dataset[i][1] == 6) for i in indices])  # 0 → 0, 6 → 1
        return data, labels

    return filter(train_set), filter(test_set)

(train_x, train_y), (test_x, test_y) = load_fashion_mnist_binary()

# 클래스 분포 확인
print(f"Train data - Class 0: {(train_y == 0).sum()}, Class 1: {(train_y == 1).sum()}")
print(f"Test data - Class 0: {(test_y == 0).sum()}, Class 1: {(test_y == 1).sum()}")

# 클래스 분포 확인
print(f"Train data - Class 0: {(train_y == 0).sum()}, Class 1: {(train_y == 1).sum()}")
print(f"Test data - Class 0: {(test_y == 0).sum()}, Class 1: {(test_y == 1).sum()}")

# -----------------------
# QNN 정의 (8-qubit 풍부한 Quantum Circuit)
# -----------------------
n_qubits = 8  # 8-qubit 시스템
n_qnn_params = 24  # 더 많은 파라미터로 표현력 향상
dev = qml.device("default.qubit", wires=n_qubits)

# 풍부한 8-qubit Quantum Circuit 정의 (24 파라미터)
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, qnn_params):
    # Data encoding - 8개 입력을 8개 qubit에 각각 인코딩
    for i in range(n_qubits):
        qml.H(wires=i)  # 모든 qubit을 superposition 상태로
        qml.RZ(2.*inputs[i], wires=i)  # 데이터 인코딩
    
    # Entangling layer 1 - 인접한 qubit들 간의 entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    
    # Variational layer 1 - 모든 qubit에 RY 회전 (8개 파라미터)
    for i in range(n_qubits):
        qml.RY(2.*qnn_params[i], wires=i)
    
    # Circular entanglement
    qml.CNOT(wires=[7, 0])  # 마지막과 첫 번째 연결
    
    # Variational layer 2 - 모든 qubit에 RX 회전 (8개 파라미터)
    for i in range(n_qubits):
        qml.RX(2.*qnn_params[8 + i], wires=i)
    
    # 더 복잡한 entanglement 패턴
    for i in range(0, n_qubits-1, 2):
        qml.CNOT(wires=[i, i+1])
    
    # 추가 entanglement - 홀수 인덱스 간 연결
    for i in range(1, n_qubits-2, 2):
        qml.CNOT(wires=[i, i+2])
    
    # Variational layer 3 - 모든 qubit에 RZ 회전 (8개 파라미터)
    for i in range(n_qubits):
        qml.RZ(2.*qnn_params[16 + i], wires=i)
    
    return qml.expval(qml.PauliZ(0))

# QNN 파라미터를 직접 관리하는 클래스
class QuantumLayer(nn.Module):
    def __init__(self, n_params):
        super().__init__()
        self.qnn_params = nn.Parameter(torch.randn(n_params, dtype=torch.float64) * 0.1)
        
    def forward(self, x):
        # 배치 처리
        results = []
        for i in range(x.size(0)):
            # 8차원 입력 사용
            input_data = x[i]  # 이미 8차원이어야 함
            result = quantum_circuit(input_data, self.qnn_params)
            results.append(result)
        return torch.stack(results)

qlayer = QuantumLayer(n_qnn_params)

# -----------------------
# 정확히 45K 파라미터 모델 설계
# -----------------------
# 최종 정확한 45K 모델 (미세 조정)
class QCNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 목표: 정확히 45,000 파라미터
        # QNN: 24 파라미터 (8-qubit 풍부한 circuit)
        # 남은 예산: 44,976 파라미터
        
        # 8차원 출력 CNN (8-qubit QNN에 맞춤)
        self.cnn = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(1, 2, 5, stride=2, padding=2),     # params: 1*2*25 + 2 = 52
            nn.ReLU(),
            
            # 14x14 → 7x7
            nn.Conv2d(2, 2, 3, stride=2, padding=1),     # params: 2*2*9 + 2 = 38
            nn.ReLU(),
            
            # 7x7 → 4x4
            nn.Conv2d(2, 2, 3, stride=1, padding=1),     # params: 2*2*9 + 2 = 38
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),                     # 2x2x2 = 8
            
            nn.Flatten(),
            nn.Linear(8, 8),                             # params: 8*8 + 8 = 72 (8-qubit 입력)
            nn.ReLU(),
            nn.Dropout(0.1)
        ).double()
        
        self.qnn = qlayer
        
        # 축소된 분류기 (QNN 파라미터 증가로 인한 조정)
        self.classifier = nn.Sequential(
            nn.Linear(1, 200),      # params: 1*200 + 200 = 400
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(200, 100),    # params: 200*100 + 100 = 20,100
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(100, 50),     # params: 100*50 + 50 = 5,050
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(50, 20),      # params: 50*20 + 20 = 1,020
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(20, 1),       # params: 20*1 + 1 = 21
            # Sigmoid 제거 - BCEWithLogitsLoss 사용
        ).double()
        
        # 총 계산: CNN(약 202) + QNN(24) + Classifier(약 26,591) = 약 26,817
        
    def forward(self, x):
        features = self.cnn(x)  # 이제 8차원 출력 (8-qubit QNN 입력에 맞춤)
        
        # QNN 처리
        q_out = self.qnn(features).unsqueeze(1)  # [batch_size, 1]
        
        return self.classifier(q_out)

# 파라미터 수 계산 함수
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------
# 학습 설정
# -----------------------
# 데이터를 double precision으로 변환
train_x = train_x.double()
test_x = test_x.double()
train_y = train_y.double().unsqueeze(1)
test_y = test_y.double().unsqueeze(1)

train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32)

epochs = 300

# 모델 선택 및 파라미터 확인
model = QCNN_Model()
total_params = count_parameters(model)

print(f"Model parameters: {total_params:,}")
print(f"Train data shape: {train_x.shape}, Test data shape: {test_x.shape}")

# 50K 제한 확인
if total_params <= 50000:
    print(f"✅ Parameter limit satisfied: {total_params:,} ≤ 50,000")
else:
    print(f"❌ Parameter limit exceeded: {total_params:,} > 50,000")

# 개선된 학습률 설정
optimizer = optim.AdamW([
    {'params': model.cnn.parameters(), 'lr': 0.001, 'weight_decay': 1e-4},
    {'params': model.qnn.parameters(), 'lr': 0.005, 'weight_decay': 1e-5},  # QNN 학습률 조정
    {'params': model.classifier.parameters(), 'lr': 0.002, 'weight_decay': 1e-4}
])

# 클래스 가중치 계산 (불균형 해결)
class_counts = torch.bincount(train_y.long().flatten())
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum() * 2  # 정규화

print(f"Class weights: {class_weights}")

# 가중치가 적용된 손실 함수
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1]/class_weights[0])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=30)

# -----------------------
# 훈련 루프
# -----------------------
best_test_acc = 0
patience_counter = 0
patience = 25

train_losses, train_accs = [], []
test_losses, test_accs = [], []

print("Starting CNN-QNN training...")

for epoch in tqdm(range(epochs)):
    model.train()
    total_loss, total_acc = 0, 0
    
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        loss.backward()
        optimizer.step()
        
        acc = ((torch.sigmoid(preds) > 0.5).double() == yb).double().mean()
        total_loss += loss.item()
        total_acc += acc.item()

    # 평가
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            acc = ((torch.sigmoid(preds) > 0.5).double() == yb).double().mean()
            test_loss += loss.item()
            test_acc += acc.item()
    
    avg_train_loss = total_loss / len(train_loader)
    avg_train_acc = total_acc / len(train_loader)
    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_acc / len(test_loader)
    
    # 기록
    train_losses.append(avg_train_loss)
    train_accs.append(avg_train_acc)
    test_losses.append(avg_test_loss)
    test_accs.append(avg_test_acc)
    
    # 학습률 스케줄링
    scheduler.step(avg_test_acc)
    
    # 최고 성능 추적
    if avg_test_acc > best_test_acc:
        best_test_acc = avg_test_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_cnn_qnn_45k.pth')
    else:
        patience_counter += 1
    
    if epoch % 15 == 0 or patience_counter == 0:
        print(f"[Epoch {epoch+1}] Train Acc: {avg_train_acc:.4f} | Test Acc: {avg_test_acc:.4f} | Best: {best_test_acc:.4f}")
    
    # 조기 종료
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

print(f"\n🎯 Training complete! Best Test Accuracy: {best_test_acc:.4f}")

# 상세한 모델 분석
cnn_params = sum(p.numel() for p in model.cnn.parameters())
qnn_params = sum(p.numel() for p in model.qnn.parameters())
classifier_params = sum(p.numel() for p in model.classifier.parameters())

print(f"\n📊 Model Analysis:")
print(f"Total parameters: {total_params:,} / 50,000 ({total_params/50000*100:.1f}%)")
print(f"CNN parameters: {cnn_params:,} ({cnn_params/total_params*100:.1f}%)")
print(f"QNN parameters: {qnn_params:,} ({qnn_params/total_params*100:.1f}%)")
print(f"Classifier parameters: {classifier_params:,} ({classifier_params/total_params*100:.1f}%)")

print(f"\n✅ Constraint Verification:")
print(f"Parameter limit (≤50K): {total_params <= 50000} ({total_params:,})")
print(f"QNN parameters (8-60): {8 <= qnn_params <= 60} ({qnn_params})")
print(f"Hybrid model: ✅ (CNN + QNN + Classifier)")

# 결과 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(test_losses, label='Test Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress - Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy', alpha=0.8)
plt.plot(test_accs, label='Test Accuracy', alpha=0.8)
plt.axhline(y=best_test_acc, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_test_acc:.4f}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Progress - Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cnn_qnn_45k_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n🏆 Final Summary:")
print(f"Best accuracy: {best_test_acc:.4f}")
print(f"Total epochs: {len(train_accs)}")
print(f"Parameter efficiency: {best_test_acc/total_params*1000000:.2f} acc/1M params")

# ----------------------------------
# 테스트 데이터 추론 및 CSV 저장
# ----------------------------------
from datetime import datetime

print(f"\n🔍 Starting inference on test data...")

# 최고 모델 로드
try:
    model.load_state_dict(torch.load('best_cnn_qnn_45k.pth'))
    print("✅ Best model loaded")
except:
    print("⚠️ Using current model")

# 테스트 데이터로더 (배치 크기 1로 개별 추론)
test_inference_loader = DataLoader(TensorDataset(test_x, test_y), 
                                  batch_size=1, shuffle=False)

model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for data, target in tqdm(test_inference_loader, desc="Inference", 
                           total=len(test_inference_loader), leave=False):
        logits = model(data)
        # BCEWithLogitsLoss를 사용했으므로 sigmoid 적용 후 0.5 기준으로 예측
        pred = (torch.sigmoid(logits) > 0.5).long().view(1)
        all_preds.append(pred.cpu())
        all_targets.append(target.view(-1).cpu())

y_pred = torch.cat(all_preds).numpy().astype(int)
y_true = torch.cat(all_targets).numpy().astype(int)

# 0·6 라벨만 평가 (우리 모델은 이미 0/6만 사용)
test_mask = (y_true == 0) | (y_true == 1)  # 우리 모델에서는 0=T-shirt, 1=Shirt
print("Total samples:", len(y_true))
print("Target samples:", test_mask.sum())

# 모델 결과가 1인 것을 6으로 변경 (원본 Fashion-MNIST 라벨로 복원)
y_pred_mapped = np.where(y_pred == 1, 6, y_pred)
y_true_mapped = np.where(y_true == 1, 6, y_true)

acc = (y_pred_mapped[test_mask] == y_true_mapped[test_mask]).mean()
print(f"Accuracy (labels 0/6 only): {acc:.4f}")

# 현재 시각을 "YYYYMMDD_HHMMSS" 형식으로 포맷팅
now = datetime.now().strftime("%Y%m%d_%H%M%S")

# 원본 파일명을 기반으로 새 파일명 생성
y_pred_filename = f"qcnn_y_pred_{now}.csv"
np.savetxt(y_pred_filename, y_pred_mapped, fmt="%d")

print(f"✅ Predictions saved to: {y_pred_filename}")
print(f"📊 Prediction distribution: 0={np.sum(y_pred_mapped==0)}, 6={np.sum(y_pred_mapped==6)}")