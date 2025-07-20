import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import pennylane as qml
from pennylane import numpy as np
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

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
# CNN → 256D 벡터로 변환
# -----------------------
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  # 28x28 → 8x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),               # 8x14x14
            nn.Conv2d(8, 16, 3, padding=1),# 16x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),               # 16x7x7
            nn.Flatten(),                  # 16*7*7 = 784
            nn.Linear(784, 256)            # → QNN input size
        )

    def forward(self, x):
        return self.encoder(x)

cnn_encoder = CNNEncoder().double()  # double precision으로 변환

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
# 전체 모델 정의
# -----------------------
class CNN_QNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNEncoder().double()  # CNN도 double precision으로 변환
        self.qnn = qlayer
        classifier = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.classifier = classifier.double()

    def forward(self, x):  # x: [B, 1, 28, 28]
        features = self.cnn(x)                      # [B, 256]
        features = torch.nn.functional.normalize(features, p=2, dim=1)  # L2 정규화
        q_out = [self.qnn(features[i]) for i in range(features.size(0))]
        q_out = torch.stack(q_out).unsqueeze(1)     # [B, 1]
        return self.classifier(q_out)

# -----------------------
# 학습 설정 (타입 통일)
# -----------------------
# 데이터를 double precision으로 변환
train_x = train_x.double()
test_x = test_x.double()
train_y = train_y.double().unsqueeze(1)
test_y = test_y.double().unsqueeze(1)

train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32)

epochs=100
model = CNN_QNN_Model()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Train data shape: {train_x.shape}, Test data shape: {test_x.shape}")

# -----------------------
# 개선된 훈련 루프
# -----------------------
best_test_acc = 0
print("Starting CNN-QNN training...")

for epoch in tqdm(range(epochs)):
    model.train()
    total_loss, total_acc = 0, 0
    
    for xb, yb in train_loader:
        # 데이터가 이미 double precision으로 변환됨
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        
        acc = (preds.round() == yb).double().mean()
        total_loss += loss.item()
        total_acc += acc.item()

    # 평가
    model.eval()
    test_acc = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            acc = (preds.round() == yb).double().mean()
            test_acc += acc.item()
    
    avg_train_acc = total_acc / len(train_loader)
    avg_test_acc = test_acc / len(test_loader)
    
    # 최고 성능 추적
    if avg_test_acc > best_test_acc:
        best_test_acc = avg_test_acc
        torch.save(model.state_dict(), 'best_cnn_qnn.pth')
    
    if epoch % 10 == 0:
        print(f"[Epoch {epoch+1}] Train Acc: {avg_train_acc:.4f} | Test Acc: {avg_test_acc:.4f} | Best: {best_test_acc:.4f}")

print(f"\n🎯 Training complete! Best Test Accuracy: {best_test_acc:.4f}")

# 모델 분석
total_params = sum(p.numel() for p in model.parameters())
cnn_params = sum(p.numel() for p in model.cnn.parameters())
qnn_params = sum(p.numel() for p in model.qnn.parameters())
classifier_params = sum(p.numel() for p in model.classifier.parameters())

print(f"\n📊 Model Analysis:")
print(f"Total parameters: {total_params:,}")
print(f"CNN parameters: {cnn_params:,}")
print(f"QNN parameters: {qnn_params:,}")
print(f"Classifier parameters: {classifier_params:,}")
print(f"QNN/Total ratio: {qnn_params/total_params:.3f}")