import torch
import torch.nn as nn
import pennylane as qml

# 8-qubit 7-parameter 시스템 확인
n_qubits = 8
n_qnn_params = 7
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev, interface='torch')
def quantum_circuit(inputs, qnn_params):
    # Data encoding
    for i in range(n_qubits):
        qml.H(wires=i)
        qml.RZ(2.*inputs[i], wires=i)
    
    # Entangling layer 1
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    
    # Variational layer 1
    qml.RY(2.*qnn_params[0], wires=0)
    qml.RY(2.*qnn_params[1], wires=2)
    qml.RY(2.*qnn_params[2], wires=4)
    qml.RY(2.*qnn_params[3], wires=6)
    
    # Circular entanglement
    qml.CNOT(wires=[7, 0])
    
    # Variational layer 2
    qml.RX(2.*qnn_params[4], wires=1)
    qml.RX(2.*qnn_params[5], wires=3)
    qml.RX(2.*qnn_params[6], wires=5)
    
    # Final entangling layer
    for i in range(0, n_qubits-1, 2):
        qml.CNOT(wires=[i, i+1])
    
    # Additional entanglement
    for i in range(1, n_qubits-2, 2):
        qml.CNOT(wires=[i, i+2])
    
    return qml.expval(qml.PauliZ(0))

class QuantumLayer(nn.Module):
    def __init__(self, n_params):
        super().__init__()
        self.qnn_params = nn.Parameter(torch.randn(n_params, dtype=torch.float64) * 0.1)
        
    def forward(self, x):
        results = []
        for i in range(x.size(0)):
            input_data = x[i]  # 8차원 입력 사용
            result = quantum_circuit(input_data, self.qnn_params)
            results.append(result)
        return torch.stack(results)

class QCNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 2, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(2, 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(8, 8),  # 8차원 출력
            nn.ReLU(),
            nn.Dropout(0.1)
        ).double()
        
        self.qnn = QuantumLayer(n_qnn_params)
        
        self.classifier = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        ).double()
        
    def forward(self, x):
        features = self.cnn(x)  # 8차원 출력
        q_out = self.qnn(features).unsqueeze(1)
        return self.classifier(q_out)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = QCNN_Model()
total_params = count_parameters(model)
cnn_params = sum(p.numel() for p in model.cnn.parameters())
qnn_params = sum(p.numel() for p in model.qnn.parameters())
classifier_params = sum(p.numel() for p in model.classifier.parameters())

print(f'Total parameters: {total_params:,}')
print(f'CNN parameters: {cnn_params:,}')
print(f'QNN parameters: {qnn_params:,}')
print(f'Classifier parameters: {classifier_params:,}')
print(f'Parameter limit check: {total_params <= 45000} (≤45K)')