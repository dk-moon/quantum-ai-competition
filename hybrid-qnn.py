import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane import numpy as np

# ----------------------------------
# 1. 사용자 정의 Fashion-MNIST 로더
# ----------------------------------
def load_fashion_mnist_binary():
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

# 데이터 로딩
(train_data, train_labels), (test_data, test_labels) = load_fashion_mnist_binary()
print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

# ----------------------------------
# 2. 개선된 데이터 전처리 (더 나은 특성 추출)
# ----------------------------------
def preprocess_data_for_amplitude_encoding(train_data, test_data):
    # (N, 1, 28, 28) → (N, 784)
    train_flat = train_data.view(train_data.size(0), -1).numpy()
    test_flat = test_data.view(test_data.size(0), -1).numpy()

    # 표준화 먼저 적용 (더 나은 PCA 성능)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_flat)
    test_scaled = scaler.transform(test_flat)

    # PCA → 256D (2^8 = 256 → 8 qubits) with higher variance retention
    pca = PCA(n_components=256, whiten=True)  # whitening 추가로 decorrelation
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

    # Normalize to unit length (L2 norm = 1)
    train_norm = normalize(train_pca)
    test_norm = normalize(test_pca)

    return torch.tensor(train_norm, dtype=torch.float64), torch.tensor(test_norm, dtype=torch.float64)

train_processed, test_processed = preprocess_data_for_amplitude_encoding(train_data, test_data)
train_labels = train_labels.double().unsqueeze(1)
test_labels = test_labels.double().unsqueeze(1)

# ----------------------------------
# 3. DataLoader
# ----------------------------------
batch_size = 32
train_loader = DataLoader(TensorDataset(train_processed, train_labels), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_processed, test_labels), batch_size=batch_size)

# ----------------------------------
# 4. 개선된 양자 회로 정의 (더 강력한 표현력)
# ----------------------------------
n_qubits = 8
n_layers = 6  # 더 효율적인 레이어 수
dev = qml.device("default.qubit", wires=n_qubits)

def enhanced_quantum_layer(weights):
    """더 강력한 양자 레이어 - 다양한 회전과 얽힘 패턴"""
    # 1. 개별 큐빗 회전 (RX, RY, RZ)
    for i in range(n_qubits):
        qml.RX(weights[i*3], wires=i)
        qml.RY(weights[i*3 + 1], wires=i)
        qml.RZ(weights[i*3 + 2], wires=i)
    
    # 2. 다양한 얽힘 패턴
    # Linear entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    
    # Circular entanglement
    qml.CNOT(wires=[n_qubits-1, 0])
    
    # All-to-all entanglement (선택적)
    for i in range(0, n_qubits, 2):
        if i + 2 < n_qubits:
            qml.CNOT(wires=[i, i+2])

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Amplitude embedding
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
    
    # Enhanced variational layers
    for layer in range(n_layers):
        layer_weights = weights[layer]
        enhanced_quantum_layer(layer_weights)
    
    # Multi-qubit measurement for richer information
    return [qml.expval(qml.PauliZ(i)) for i in range(min(4, n_qubits))]  # 4개 큐빗 측정

weight_shapes = {"weights": (n_layers, n_qubits * 3)}  # RX, RY, RZ per qubit
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

print(f"Total quantum parameters: {n_layers * n_qubits * 3}")  # 파라미터 수 확인

# ----------------------------------
# 5. 개선된 하이브리드 모델 (더 강력한 고전 부분)
# ----------------------------------
class EnhancedHybridQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qlayer
        
        # 더 강력한 고전 신경망 (4개 측정값 입력)
        self.classifier = nn.Sequential(
            nn.Linear(4, 32),  # 4개 측정값 입력
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(16, 8),
            nn.ReLU(),
            
            nn.Linear(8, 1),
            nn.Sigmoid()
        ).double()
        
        # 추가: 잔차 연결을 위한 직접 특성 처리
        self.feature_processor = nn.Sequential(
            nn.Linear(256, 64),  # PCA 출력 직접 처리
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Tanh()
        ).double()
        
        # 양자 출력 차원 조정을 위한 레이어
        self.quantum_adapter = nn.Linear(4, 4).double()

    def forward(self, x):
        # 양자 부분
        q_out = []
        for i in range(x.shape[0]):
            q_result = self.qlayer(x[i])
            # q_result가 이미 텐서인 경우와 리스트인 경우 모두 처리
            if isinstance(q_result, (list, tuple)):
                q_tensor = torch.stack(q_result)
            else:
                # 단일 측정값인 경우 4개로 복제하여 차원 맞춤
                q_tensor = q_result.repeat(4) if q_result.dim() == 0 else q_result
            q_out.append(q_tensor)
        
        q_out = torch.stack(q_out)
        
        # 양자 출력 차원 조정
        q_features = self.quantum_adapter(q_out)
        
        # 고전 특성 추출
        classical_features = self.feature_processor(x)
        
        # 양자-고전 특성 결합 (잔차 연결)
        combined_features = q_features + classical_features
        
        return self.classifier(combined_features)

model = EnhancedHybridQNN()
criterion = nn.BCELoss()

# 개선된 옵티마이저 설정 (다른 학습률)
quantum_params = list(model.qlayer.parameters())
classical_params = list(model.classifier.parameters()) + list(model.feature_processor.parameters())

optimizer = optim.AdamW([
    {'params': quantum_params, 'lr': 0.005, 'weight_decay': 1e-4},  # 양자 부분은 낮은 학습률
    {'params': classical_params, 'lr': 0.01, 'weight_decay': 1e-3}  # 고전 부분은 높은 학습률
])

# 학습률 스케줄러 추가
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=10)

print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")

# ----------------------------------
# 6. 개선된 훈련 루프 (조기 종료, 모니터링, 정규화)
# ----------------------------------
epochs = 150
best_test_acc = 0
patience_counter = 0
patience = 20

train_losses, train_accs = [], []
test_losses, test_accs = [], []

print("Starting enhanced training...")
for epoch in range(epochs):
    # 훈련
    model.train()
    total_loss, total_acc = 0, 0

    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        optimizer.zero_grad()
        pred = model(xb)
        
        # Focal Loss for better class balance (optional enhancement)
        loss = criterion(pred, yb)
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        loss.backward()
        optimizer.step()
        
        acc = (pred.round() == yb).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()

    # 평가
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            acc = (pred.round() == yb).float().mean()
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
    
    # 조기 종료 체크
    if avg_test_acc > best_test_acc:
        best_test_acc = avg_test_acc
        patience_counter = 0
        # 최고 모델 저장
        torch.save(model.state_dict(), 'best_hybrid_qnn.pth')
    else:
        patience_counter += 1
    
    if epoch % 10 == 0 or patience_counter == 0:
        print(f"[Epoch {epoch+1}] Train Acc: {avg_train_acc:.4f} | Test Acc: {avg_test_acc:.4f} | Best: {best_test_acc:.4f}")
    
    # 조기 종료
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

print(f"Training complete ✅ Best Test Accuracy: {best_test_acc:.4f}")

# ----------------------------------
# 7. 결과 시각화 및 분석
# ----------------------------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(test_losses, label='Test Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(train_accs, label='Train Accuracy', alpha=0.8)
plt.plot(test_accs, label='Test Accuracy', alpha=0.8)
plt.axhline(y=best_test_acc, color='r', linestyle='--', alpha=0.7, label=f'Best Test: {best_test_acc:.4f}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 학습률 변화 시각화
plt.subplot(1, 3, 3)
lrs = [group['lr'] for group in optimizer.param_groups]
plt.bar(['Quantum LR', 'Classical LR'], lrs, alpha=0.7, color=['blue', 'orange'])
plt.ylabel('Learning Rate')
plt.title('Current Learning Rates')
plt.yscale('log')

plt.tight_layout()
plt.savefig('hybrid_qnn_training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 최고 모델 로드 및 최종 평가
try:
    model.load_state_dict(torch.load('best_hybrid_qnn.pth'))
    print("✅ Best model loaded successfully")
except:
    print("⚠️ Using current model (best model file not found)")

model.eval()

# 상세한 최종 평가
final_test_acc = 0
all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb)
        acc = (pred.round() == yb).float().mean()
        final_test_acc += acc.item()
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

final_test_acc /= len(test_loader)

# 분류 성능 분석
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
pred_classes = (all_preds > 0.5).astype(int)

print(f"\n🎯 Final Test Accuracy: {final_test_acc:.4f}")
print(f"\n📊 Classification Report:")
print(classification_report(all_labels, pred_classes, target_names=['T-shirt/top', 'Shirt']))

print(f"\n🔍 Confusion Matrix:")
cm = confusion_matrix(all_labels, pred_classes)
print(cm)

# 모델 복잡도 분석
total_params = sum(p.numel() for p in model.parameters())
quantum_params_count = sum(p.numel() for p in model.qlayer.parameters())
classical_params_count = total_params - quantum_params_count

print(f"\n📈 Model Analysis:")
print(f"Total parameters: {total_params:,}")
print(f"Quantum parameters: {quantum_params_count:,}")
print(f"Classical parameters: {classical_params_count:,}")
print(f"Quantum/Classical ratio: {quantum_params_count/classical_params_count:.3f}")

# 양자 회로 깊이 분석
circuit_depth = n_layers * (3 + 3)  # 3 rotations + 3 entanglement operations per layer
print(f"Estimated circuit depth: {circuit_depth}")
print(f"Circuit efficiency (acc/depth): {final_test_acc/circuit_depth:.6f}")

print(f"\n🏆 Training Summary:")
print(f"Best accuracy achieved: {best_test_acc:.4f}")
print(f"Final accuracy: {final_test_acc:.4f}")
print(f"Total epochs trained: {len(train_accs)}")
print(f"Improvement over baseline: {(final_test_acc - 0.5) * 100:.2f}% above random")