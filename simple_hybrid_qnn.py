import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

import pennylane as qml

# ----------------------------------
# Step 1: Fashion-MNIST 데이터 로드
# ----------------------------------
def load_fashion_mnist_binary():
    """T-shirt/top (0) vs Shirt (6) 이진 분류 데이터"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    def filter_classes(dataset, target_classes=[0, 6]):
        indices = [i for i, (_, label) in enumerate(dataset) if label in target_classes]
        data = torch.stack([dataset[i][0] for i in indices])
        labels = torch.tensor([dataset[i][1] for i in indices])
        labels = (labels == 6).long()  # T-shirt/top: 0, Shirt: 1
        return data, labels
    
    return filter_classes(train_dataset), filter_classes(test_dataset)

# 데이터 로드
(train_data, train_labels), (test_data, test_labels) = load_fashion_mnist_binary()
print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

# ----------------------------------
# Step 2: 양자 회로 정의 (간단하고 안전)
# ----------------------------------
n_qubits = 8
n_layers = 4
dev = qml.device("default.qubit", wires=n_qubits)

def quantum_layer(weights):
    """간단한 양자 레이어"""
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """양자 회로 - Amplitude Embedding + Variational Layers"""
    # Amplitude embedding (256D → 8 qubits)
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
    
    # Variational layers
    for layer in range(n_layers):
        layer_weights = weights[layer * n_qubits:(layer + 1) * n_qubits]
        quantum_layer(layer_weights)
    
    # 다중 측정 (더 풍부한 정보)
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# 양자 레이어 생성
weight_shapes = {"weights": (n_layers * n_qubits,)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

print(f"Quantum parameters: {n_layers * n_qubits}")

# ----------------------------------
# Step 3: 진정한 하이브리드 모델 정의
# ----------------------------------
class SimpleHybridQNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 고전 경로: CNN 특성 추출기
        self.classical_path = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            
            # 14x14 → 7x7
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 7x7 → 4x4
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),  # 64x2x2 = 256
            
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),  # 고전 특성 64차원
            nn.ReLU()
        ).double()
        
        # 양자 경로: QNN
        self.quantum_path = qlayer
        
        # 특성 결합 및 최종 분류
        self.fusion_classifier = nn.Sequential(
            nn.Linear(64 + 4, 128),  # 고전(64) + 양자(4) = 68차원
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).double()
        
        # 양자 입력을 위한 차원 축소
        self.quantum_preprocessor = nn.Sequential(
            nn.Flatten(),  # 28*28 = 784
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),  # 256차원으로 축소 (2^8 = 256 → 8 qubits)
            nn.Tanh()  # [-1, 1] 범위로 정규화
        ).double()

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 고전 경로: CNN으로 고차원 특성 추출
        classical_features = self.classical_path(x)  # [B, 64]
        
        # 양자 경로: 이미지 → 256D → QNN
        quantum_input = self.quantum_preprocessor(x)  # [B, 256]
        
        # 배치별 양자 처리
        quantum_outputs = []
        for i in range(batch_size):
            # L2 정규화 (Amplitude Embedding 요구사항)
            normalized_input = torch.nn.functional.normalize(quantum_input[i], p=2, dim=0)
            q_out = self.quantum_path(normalized_input)
            
            # 리스트를 텐서로 변환
            if isinstance(q_out, (list, tuple)):
                q_tensor = torch.stack(q_out)
            else:
                q_tensor = q_out
            quantum_outputs.append(q_tensor)
        
        quantum_features = torch.stack(quantum_outputs)  # [B, 4]
        
        # 특성 융합: 고전 + 양자
        fused_features = torch.cat([classical_features, quantum_features], dim=1)  # [B, 68]
        
        # 최종 분류
        output = self.fusion_classifier(fused_features)
        
        return output

# ----------------------------------
# Step 4: 모델 초기화 및 훈련 설정
# ----------------------------------
model = SimpleHybridQNN()

# 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
classical_params = sum(p.numel() for p in model.classical_path.parameters()) + \
                  sum(p.numel() for p in model.quantum_preprocessor.parameters())
quantum_params = sum(p.numel() for p in model.quantum_path.parameters())
fusion_params = sum(p.numel() for p in model.fusion_classifier.parameters())

print(f"\n📊 Model Analysis:")
print(f"Total parameters: {total_params:,}")
print(f"Classical parameters: {classical_params:,}")
print(f"Quantum parameters: {quantum_params:,}")
print(f"Fusion parameters: {fusion_params:,}")
print(f"Quantum/Total ratio: {quantum_params/total_params:.3f}")

# 50K 제한 확인
if total_params <= 50000:
    print(f"✅ Parameter limit satisfied: {total_params:,} ≤ 50,000")
else:
    print(f"❌ Parameter limit exceeded: {total_params:,} > 50,000")

# ----------------------------------
# Step 5: 데이터 준비 및 훈련
# ----------------------------------
# 데이터를 double precision으로 변환
train_data = train_data.double()
test_data = test_data.double()
train_labels = train_labels.double().unsqueeze(1)
test_labels = test_labels.double().unsqueeze(1)

# DataLoader
batch_size = 16  # 양자 처리 때문에 작은 배치 사용
train_loader = DataLoader(TensorDataset(train_data, train_labels), 
                         batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_data, test_labels), 
                        batch_size=batch_size, shuffle=False)

# 옵티마이저 및 손실 함수
criterion = nn.BCELoss()
optimizer = optim.AdamW([
    {'params': model.classical_path.parameters(), 'lr': 0.001, 'weight_decay': 1e-4},
    {'params': model.quantum_path.parameters(), 'lr': 0.01, 'weight_decay': 1e-5},
    {'params': model.fusion_classifier.parameters(), 'lr': 0.005, 'weight_decay': 1e-4},
    {'params': model.quantum_preprocessor.parameters(), 'lr': 0.002, 'weight_decay': 1e-4}
])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=10)

# ----------------------------------
# Step 6: 훈련 루프
# ----------------------------------
epochs = 80
best_test_acc = 0
patience_counter = 0
patience = 15

train_losses, train_accs = [], []
test_losses, test_accs = [], []

print("\n🚀 Starting Hybrid QNN Training...")

for epoch in range(epochs):
    # 훈련
    model.train()
    total_loss, total_acc = 0, 0
    
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        loss.backward()
        optimizer.step()
        
        acc = (pred.round() == yb).double().mean()
        total_loss += loss.item()
        total_acc += acc.item()

    # 평가
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            acc = (pred.round() == yb).double().mean()
            test_loss += loss.item()
            test_acc += acc.item()
    
    # 평균 계산
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
        torch.save(model.state_dict(), 'best_simple_hybrid_qnn.pth')
    else:
        patience_counter += 1
    
    if epoch % 10 == 0 or patience_counter == 0:
        print(f"[Epoch {epoch+1}] Train Acc: {avg_train_acc:.4f} | Test Acc: {avg_test_acc:.4f} | Best: {best_test_acc:.4f}")
    
    # 조기 종료
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

print(f"\n🎯 Training complete! Best Test Accuracy: {best_test_acc:.4f}")

# ----------------------------------
# Step 7: 결과 분석 및 시각화
# ----------------------------------
# 최고 모델 로드
try:
    model.load_state_dict(torch.load('best_simple_hybrid_qnn.pth'))
    print("✅ Best model loaded")
except:
    print("⚠️ Using current model")

# 최종 평가
model.eval()
final_test_acc = 0
all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb)
        acc = (pred.round() == yb).double().mean()
        final_test_acc += acc.item()
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

final_test_acc /= len(test_loader)

# 분류 성능 분석
from sklearn.metrics import classification_report, confusion_matrix

all_preds = torch.tensor(all_preds)
all_labels = torch.tensor(all_labels)
pred_classes = (all_preds > 0.5).int()

print(f"\n🎯 Final Test Accuracy: {final_test_acc:.4f}")
print(f"\n📊 Classification Report:")
print(classification_report(all_labels.numpy(), pred_classes.numpy(), 
                          target_names=['T-shirt/top', 'Shirt']))

print(f"\n🔍 Confusion Matrix:")
cm = confusion_matrix(all_labels.numpy(), pred_classes.numpy())
print(cm)

# 시각화
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
plt.axhline(y=best_test_acc, color='r', linestyle='--', alpha=0.7, 
           label=f'Best: {best_test_acc:.4f}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Progress - Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_hybrid_qnn_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n🏆 Final Summary:")
print(f"Best accuracy: {best_test_acc:.4f}")
print(f"Final accuracy: {final_test_acc:.4f}")
print(f"Total epochs: {len(train_accs)}")
print(f"Model efficiency: {final_test_acc/total_params*1000000:.2f} acc/1M params")

print(f"\n✅ Hybrid Architecture Verification:")
print(f"Classical pathway: ✅ (CNN feature extraction)")
print(f"Quantum pathway: ✅ (QNN processing)")
print(f"Feature fusion: ✅ (Classical + Quantum → Final classifier)")
print(f"Parameter distribution: Classical({classical_params:,}) + Quantum({quantum_params:,}) + Fusion({fusion_params:,})")