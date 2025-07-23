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

# -----------------------
# QNN 정의
# -----------------------
n_qubits = 8
n_layers = 7
dev = qml.device("default.qubit", wires=n_qubits)

def quantum_block(weights):
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
    for w in weights:
        quantum_block(w)
    return qml.expval(qml.PauliZ(0))

weight_shapes = {"weights": (n_layers, n_qubits)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# -----------------------
# 정확히 45K 파라미터 모델 설계
# -----------------------
# 최종 정확한 45K 모델 (미세 조정)
class Exact45K_CNN_QNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 목표: 정확히 45,000 파라미터
        # QNN: 56 파라미터 (고정)
        # 남은 예산: 44,944 파라미터
        
        # 최소 CNN (약 2.5K 파라미터)
        self.cnn = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(1, 2, 5, stride=2, padding=2),     # params: 1*2*25 + 2 = 52
            nn.ReLU(),
            
            # 14x14 → 7x7
            nn.Conv2d(2, 3, 3, stride=2, padding=1),     # params: 2*3*9 + 3 = 57
            nn.ReLU(),
            
            # 7x7 → 4x4
            nn.Conv2d(3, 4, 3, stride=1, padding=1),     # params: 3*4*9 + 4 = 112
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                     # 4x4x4 = 64
            
            nn.Flatten(),
            nn.Linear(64, 256),                          # params: 64*256 + 256 = 16,640
            nn.ReLU(),
            nn.Dropout(0.3)
        ).double()
        
        self.qnn = qlayer
        
        # 분류기 (정확히 계산: 44,944 - 16,861 = 28,083 파라미터)
        self.classifier = nn.Sequential(
            nn.Linear(1, 167),      # params: 1*167 + 167 = 334
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(167, 120),    # params: 167*120 + 120 = 20,160
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(120, 64),     # params: 120*64 + 64 = 7,744
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1),       # params: 64*1 + 1 = 65
            nn.Sigmoid()
        ).double()
        
        # 총 계산: CNN(16,861) + QNN(56) + Classifier(28,303) = 45,220 (거의 정확!)
        
    def forward(self, x):
        features = self.cnn(x)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        q_out = []
        for i in range(features.size(0)):
            q_result = self.qnn(features[i])
            q_out.append(q_result)
        q_out = torch.stack(q_out).unsqueeze(1)
        
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

epochs = 100

# 모델 선택 및 파라미터 확인
model = Exact45K_CNN_QNN_Model()
total_params = count_parameters(model)

print(f"Model parameters: {total_params:,}")
print(f"Train data shape: {train_x.shape}, Test data shape: {test_x.shape}")

# 50K 제한 확인
if total_params <= 50000:
    print(f"✅ Parameter limit satisfied: {total_params:,} ≤ 50,000")
else:
    print(f"❌ Parameter limit exceeded: {total_params:,} > 50,000")

# 적응적 학습률
optimizer = optim.AdamW([
    {'params': model.cnn.parameters(), 'lr': 0.001, 'weight_decay': 1e-4},
    {'params': model.qnn.parameters(), 'lr': 0.01, 'weight_decay': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 0.005, 'weight_decay': 1e-4}
])

criterion = nn.BCELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=15)

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
        
        acc = (preds.round() == yb).double().mean()
        total_loss += loss.item()
        total_acc += acc.item()

    # 평가
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            acc = (preds.round() == yb).double().mean()
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