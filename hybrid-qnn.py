import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

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
# 2. PCA + 정규화 (Amplitude Encoding 준비)
# ----------------------------------
def preprocess_data_for_amplitude_encoding(train_data, test_data):
    # (N, 1, 28, 28) → (N, 784)
    train_flat = train_data.view(train_data.size(0), -1).numpy()
    test_flat = test_data.view(test_data.size(0), -1).numpy()

    # PCA → 256D (2^8 = 256 → 8 qubits)
    pca = PCA(n_components=256)
    train_pca = pca.fit_transform(train_flat)
    test_pca = pca.transform(test_flat)

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
# 4. 양자 회로 정의 (Amplitude Encoding)
# ----------------------------------
n_qubits = 8
n_layers = 7  # 7 × 8 = 56 params < 60
qml.device("default.qubit", wires=n_qubits)

dev = qml.device("default.qubit", wires=n_qubits)

def quantum_layer(weights):
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
    for w in weights:
        quantum_layer(w)
    return qml.expval(qml.PauliZ(0))

weight_shapes = {"weights": (n_layers, n_qubits)}
qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# ----------------------------------
# 5. 전체 모델 정의 (Quantum + Classical)
# ----------------------------------
class HybridQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qlayer
        self.classifier = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).double()

    def forward(self, x):
        q_out = [self.qlayer(x[i]) for i in range(x.shape[0])]
        q_out = torch.stack(q_out).unsqueeze(1)
        return self.classifier(q_out)

model = HybridQNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ----------------------------------
# 6. 훈련 루프
# ----------------------------------
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss, total_acc = 0, 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        
        acc = (pred.round() == yb).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()

    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            acc = (pred.round() == yb).float().mean()
            test_loss += loss.item()
            test_acc += acc.item()

    print(f"[Epoch {epoch+1}] Train Acc: {total_acc/len(train_loader):.4f} | Test Acc: {test_acc/len(test_loader):.4f}")

print("Training complete ✅")