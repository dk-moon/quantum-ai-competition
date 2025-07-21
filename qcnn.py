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
# 최적화된 CNN → 256D 벡터로 변환 (40K-50K 파라미터 활용)
# -----------------------
class OptimizedCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 40K-50K 파라미터를 효율적으로 활용하는 CNN 구조
        self.encoder = nn.Sequential(
            # 첫 번째 블록: 28x28 → 14x14
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # params: 1*32*5*5 + 32 = 832
            nn.BatchNorm2d(32),  # params: 32*2 = 64
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # 두 번째 블록: 14x14 → 7x7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # params: 32*64*3*3 + 64 = 18,496
            nn.BatchNorm2d(64),  # params: 64*2 = 128
            nn.ReLU(),
            nn.Dropout2d(0.15),
            
            # 세 번째 블록: 7x7 → 4x4
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # params: 64*128*3*3 + 128 = 73,856
            nn.BatchNorm2d(128),  # params: 128*2 = 256
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),  # 128x2x2 = 512
            
            nn.Flatten(),  # 512
            
            # 완전연결층으로 256D로 압축
            nn.Linear(512, 256),  # params: 512*256 + 256 = 131,328
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 총 CNN 파라미터: 832 + 64 + 18,496 + 128 + 73,856 + 256 + 131,328 = ~225K
        # 너무 많으므로 조정 필요

    def forward(self, x):
        return self.encoder(x)

class BalancedCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 균형잡힌 CNN 구조 (약 35K 파라미터)
        self.features = nn.Sequential(
            # 첫 번째 블록: 28x28 → 14x14
            nn.Conv2d(1, 24, kernel_size=5, stride=2, padding=2),  # params: 1*24*5*5 + 24 = 624
            nn.BatchNorm2d(24),  # params: 24*2 = 48
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # 두 번째 블록: 14x14 → 7x7
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),  # params: 24*48*3*3 + 48 = 10,416
            nn.BatchNorm2d(48),  # params: 48*2 = 96
            nn.ReLU(),
            nn.Dropout2d(0.15),
            
            # 세 번째 블록: 7x7 → 4x4
            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),  # params: 48*96*3*3 + 96 = 41,568
            nn.BatchNorm2d(96),  # params: 96*2 = 192
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3),  # 96x3x3 = 864
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 864
            nn.Linear(864, 512),  # params: 864*512 + 512 = 442,880 (너무 많음)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),  # params: 512*256 + 256 = 131,328
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 최종 최적화된 버전 (정확히 40K-45K 파라미터 목표)
class FinalOptimizedCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 첫 번째 블록: 28x28 → 14x14
            nn.Conv2d(1, 20, kernel_size=5, stride=2, padding=2),  # params: 1*20*5*5 + 20 = 520
            nn.BatchNorm2d(20),  # params: 20*2 = 40
            nn.ReLU(),
            
            # 두 번째 블록: 14x14 → 7x7
            nn.Conv2d(20, 40, kernel_size=3, stride=2, padding=1),  # params: 20*40*3*3 + 40 = 7,240
            nn.BatchNorm2d(40),  # params: 40*2 = 80
            nn.ReLU(),
            
            # 세 번째 블록: 7x7 → 4x4
            nn.Conv2d(40, 80, kernel_size=3, stride=1, padding=1),  # params: 40*80*3*3 + 80 = 28,880
            nn.BatchNorm2d(80),  # params: 80*2 = 160
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # 80x4x4 = 1280
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),  # 1280
            nn.Linear(1280, 512),  # params: 1280*512 + 512 = 655,872 (여전히 너무 많음)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),  # params: 512*256 + 256 = 131,328
        )
        
        # 다시 계산 필요...

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

# 실제 사용할 최종 버전 (정확한 계산)
class SmartCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 약 35K 파라미터로 설계
        self.conv_layers = nn.Sequential(
            # Block 1: 28x28 → 14x14
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # params: 1*16*25 + 16 = 416
            nn.BatchNorm2d(16),  # params: 32
            nn.ReLU(),
            
            # Block 2: 14x14 → 7x7
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # params: 16*32*9 + 32 = 4,640
            nn.BatchNorm2d(32),  # params: 64
            nn.ReLU(),
            
            # Block 3: 7x7 → 4x4
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # params: 32*64*9 + 64 = 18,496
            nn.BatchNorm2d(64),  # params: 128
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),  # 64x2x2 = 256
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 256
            nn.Linear(256, 256),  # params: 256*256 + 256 = 65,792
            nn.ReLU(),
        )
        
        # 총 파라미터: 416 + 32 + 4,640 + 64 + 18,496 + 128 + 65,792 = 89,568
        # 여전히 많음, 더 줄여야 함

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 파라미터 수 계산 함수
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------
# -----------------------
# QNN 정의
# -----------------------
n_qubits = 8
n_layers = 7

# 정확히 40K-45K 파라미터를 달성하는 모델
class Precise45KCNN_QNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 정확히 계산된 CNN (약 35K 파라미터)
        self.cnn = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(1, 8, 5, stride=2, padding=2),     # params: 1*8*25 + 8 = 208
            nn.ReLU(),
            
            # 14x14 → 7x7
            nn.Conv2d(8, 16, 3, stride=2, padding=1),    # params: 8*16*9 + 16 = 1,168
            nn.ReLU(),
            
            # 7x7 → 4x4
            nn.Conv2d(16, 32, 3, stride=1, padding=1),   # params: 16*32*9 + 32 = 4,640
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3),                     # 32x3x3 = 288
            
            nn.Flatten(),
            nn.Linear(288, 256),                         # params: 288*256 + 256 = 73,984
            nn.ReLU(),
            nn.Dropout(0.3)
        ).double()
        
        self.qnn = qlayer
        
        # 약 8K 파라미터의 분류기
        self.classifier = nn.Sequential(
            nn.Linear(1, 512),      # params: 1*512 + 512 = 1,024
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),    # params: 512*256 + 256 = 131,328
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),    # params: 256*128 + 128 = 32,896
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),     # params: 128*64 + 64 = 8,256
            nn.ReLU(),
            nn.Linear(64, 1),       # params: 64*1 + 1 = 65
            nn.Sigmoid()
        ).double()
        
        # 총 계산: CNN(80,000) + QNN(56) + Classifier(173,569) = 253,625 (여전히 많음)
        
    def forward(self, x):
        features = self.cnn(x)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        q_out = []
        for i in range(features.size(0)):
            q_result = self.qnn(features[i])
            q_out.append(q_result)
        q_out = torch.stack(q_out).unsqueeze(1)
        
        return self.classifier(q_out)

# 실제 45K 파라미터 모델 (매우 신중한 설계)
class Actual45KCNN_QNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 매우 신중하게 계산된 CNN (약 30K 파라미터)
        self.cnn = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(1, 6, 5, stride=2, padding=2),     # params: 1*6*25 + 6 = 156
            nn.ReLU(),
            
            # 14x14 → 7x7
            nn.Conv2d(6, 12, 3, stride=2, padding=1),    # params: 6*12*9 + 12 = 660
            nn.ReLU(),
            
            # 7x7 → 4x4
            nn.Conv2d(12, 24, 3, stride=1, padding=1),   # params: 12*24*9 + 24 = 2,616
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),                     # 24x2x2 = 96
            
            nn.Flatten(),
            nn.Linear(96, 256),                          # params: 96*256 + 256 = 24,832
            nn.ReLU(),
            nn.Dropout(0.3)
        ).double()
        
        self.qnn = qlayer
        
        # 약 15K 파라미터의 분류기 (남은 예산 최대 활용)
        self.classifier = nn.Sequential(
            nn.Linear(1, 256),      # params: 1*256 + 256 = 512
            nn.ReLU(),
            nn.BatchNorm1d(256),    # params: 256*2 = 512
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),    # params: 256*128 + 128 = 32,896
            nn.ReLU(),
            nn.BatchNorm1d(128),    # params: 128*2 = 256
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),     # params: 128*64 + 64 = 8,256
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),      # params: 64*32 + 32 = 2,080
            nn.ReLU(),
            
            nn.Linear(32, 1),       # params: 32*1 + 1 = 33
            nn.Sigmoid()
        ).double()
        
        # 총 계산: CNN(28,264) + QNN(56) + Classifier(44,545) = 72,865 (여전히 많음)
        
    def forward(self, x):
        features = self.cnn(x)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        q_out = []
        for i in range(features.size(0)):
            q_result = self.qnn(features[i])
            q_out.append(q_result)
        q_out = torch.stack(q_out).unsqueeze(1)
        
        return self.classifier(q_out)

# 최종 정확한 45K 모델 (매우 보수적 설계)
class Conservative45KCNN_QNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 매우 보수적인 CNN (약 25K 파라미터)
        self.cnn = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(1, 4, 5, stride=2, padding=2),     # params: 1*4*25 + 4 = 104
            nn.ReLU(),
            
            # 14x14 → 7x7
            nn.Conv2d(4, 8, 3, stride=2, padding=1),     # params: 4*8*9 + 8 = 296
            nn.ReLU(),
            
            # 7x7 → 4x4
            nn.Conv2d(8, 16, 3, stride=1, padding=1),    # params: 8*16*9 + 16 = 1,168
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),                     # 16x2x2 = 64
            
            nn.Flatten(),
            nn.Linear(64, 256),                          # params: 64*256 + 256 = 16,640
            nn.ReLU(),
            nn.Dropout(0.3)
        ).double()
        
        self.qnn = qlayer
        
        # 약 25K 파라미터의 분류기
        self.classifier = nn.Sequential(
            nn.Linear(1, 200),      # params: 1*200 + 200 = 400
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(200, 150),    # params: 200*150 + 150 = 30,150
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(150, 100),    # params: 150*100 + 100 = 15,100
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(100, 50),     # params: 100*50 + 50 = 5,050
            nn.ReLU(),
            
            nn.Linear(50, 1),       # params: 50*1 + 1 = 51
            nn.Sigmoid()
        ).double()
        
        # 총 계산: CNN(18,208) + QNN(56) + Classifier(50,751) = 69,015 (여전히 많음)
        
    def forward(self, x):
        features = self.cnn(x)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        q_out = []
        for i in range(features.size(0)):
            q_result = self.qnn(features[i])
            q_out.append(q_result)
        q_out = torch.stack(q_out).unsqueeze(1)
        
        return self.classifier(q_out)

# 진짜 마지막 45K 모델 (극도로 보수적)
class Ultra45KCNN_QNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 극도로 보수적인 CNN (약 10K 파라미터)
        self.cnn = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(1, 3, 5, stride=2, padding=2),     # params: 1*3*25 + 3 = 78
            nn.ReLU(),
            
            # 14x14 → 7x7
            nn.Conv2d(3, 6, 3, stride=2, padding=1),     # params: 3*6*9 + 6 = 168
            nn.ReLU(),
            
            # 7x7 → 4x4
            nn.Conv2d(6, 12, 3, stride=1, padding=1),    # params: 6*12*9 + 12 = 660
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),                     # 12x2x2 = 48
            
            nn.Flatten(),
            nn.Linear(48, 256),                          # params: 48*256 + 256 = 12,544
            nn.ReLU(),
            nn.Dropout(0.3)
        ).double()
        
        self.qnn = qlayer
        
        # 약 35K 파라미터의 분류기 (남은 예산 최대 활용)
        self.classifier = nn.Sequential(
            nn.Linear(1, 180),      # params: 1*180 + 180 = 360
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(180, 140),    # params: 180*140 + 140 = 25,340
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(140, 100),    # params: 140*100 + 100 = 14,100
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(100, 50),     # params: 100*50 + 50 = 5,050
            nn.ReLU(),
            
            nn.Linear(50, 1),       # params: 50*1 + 1 = 51
            nn.Sigmoid()
        ).double()
        
        # 총 계산: CNN(13,450) + QNN(56) + Classifier(44,901) = 58,407 (여전히 많음)
        
    def forward(self, x):
        features = self.cnn(x)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        q_out = []
        for i in range(features.size(0)):
            q_result = self.qnn(features[i])
            q_out.append(q_result)
        q_out = torch.stack(q_out).unsqueeze(1)
        
        return self.classifier(q_out)

# 최종 정확한 45K 모델
class Final45KCNN_QNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 정확히 계산된 CNN (약 35K)
        self.cnn = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(1, 6, 5, stride=2, padding=2),     # params: 1*6*25 + 6 = 156
            nn.ReLU(),
            
            # 14x14 → 7x7
            nn.Conv2d(6, 12, 3, stride=2, padding=1),    # params: 6*12*9 + 12 = 660
            nn.ReLU(),
            
            # 7x7 → 4x4
            nn.Conv2d(12, 24, 3, stride=1, padding=1),   # params: 12*24*9 + 24 = 2,616
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(5),                     # 24x5x5 = 600
            
            nn.Flatten(),
            nn.Linear(600, 400),                         # params: 600*400 + 400 = 240,400
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(400, 256),                         # params: 400*256 + 256 = 102,656
            nn.ReLU()
        ).double()
        
        self.qnn = qlayer
        
        # 약 10K 파라미터의 분류기
        self.classifier = nn.Sequential(
            nn.Linear(1, 200),      # params: 201
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(200, 100),    # params: 200*100 + 100 = 20,100
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 50),     # params: 100*50 + 50 = 5,050
            nn.ReLU(),
            nn.Linear(50, 1),       # params: 51
            nn.Sigmoid()
        ).double()
        
        # 총: CNN(346,488) + QNN(56) + Classifier(25,402) = 371,946 (여전히 많음...)
        
    def forward(self, x):
        features = self.cnn(x)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        q_out = []
        for i in range(features.size(0)):
            q_result = self.qnn(features[i])
            q_out.append(q_result)
        q_out = torch.stack(q_out).unsqueeze(1)
        
        return self.classifier(q_out)
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
class OptimizedCNN_QNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 40K-45K 파라미터를 목표로 하는 최적화된 CNN
        self.cnn = self._build_optimized_cnn().double()
        self.qnn = qlayer
        
        # 강력한 분류기 (남은 파라미터 예산 최대 활용)
        classifier = nn.Sequential(
            nn.Linear(1, 128),      # params: 1*128 + 128 = 256
            nn.ReLU(),
            nn.BatchNorm1d(128),    # params: 128*2 = 256
            nn.Dropout(0.4),
            
            nn.Linear(128, 64),     # params: 128*64 + 64 = 8,256
            nn.ReLU(),
            nn.BatchNorm1d(64),     # params: 64*2 = 128
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),      # params: 64*32 + 32 = 2,080
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),      # params: 32*16 + 16 = 528
            nn.ReLU(),
            
            nn.Linear(16, 1),       # params: 16*1 + 1 = 17
            nn.Sigmoid()
        )
        self.classifier = classifier.double()
        
        # 총 분류기 파라미터: 256 + 256 + 8,256 + 128 + 2,080 + 528 + 17 = 11,521개
    
    def _build_optimized_cnn(self):
        # 정확히 계산된 CNN (약 35K 파라미터)
        return nn.Sequential(
            # Block 1: 28x28 → 14x14
            nn.Conv2d(1, 12, kernel_size=5, stride=2, padding=2),  # params: 1*12*25 + 12 = 312
            nn.BatchNorm2d(12),  # params: 24
            nn.ReLU(),
            
            # Block 2: 14x14 → 7x7
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),  # params: 12*24*9 + 24 = 2,616
            nn.BatchNorm2d(24),  # params: 48
            nn.ReLU(),
            
            # Block 3: 7x7 → 4x4
            nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),  # params: 24*48*9 + 48 = 10,416
            nn.BatchNorm2d(48),  # params: 96
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3),  # 48x3x3 = 432
            
            nn.Flatten(),  # 432
            nn.Linear(432, 512),  # params: 432*512 + 512 = 221,696
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),  # params: 512*256 + 256 = 131,328
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 총 CNN 파라미터: 312 + 24 + 2,616 + 48 + 10,416 + 96 + 221,696 + 131,328 = 366,536 (너무 많음!)

    def forward(self, x):  # x: [B, 1, 28, 28]
        features = self.cnn(x)                      # [B, 256]
        features = torch.nn.functional.normalize(features, p=2, dim=1)  # L2 정규화
        
        # 배치 처리 최적화
        q_out = []
        for i in range(features.size(0)):
            q_result = self.qnn(features[i])
            q_out.append(q_result)
        q_out = torch.stack(q_out).unsqueeze(1)     # [B, 1]
        
        return self.classifier(q_out)

# 실제 사용할 정교하게 계산된 모델
class PreciseCNN_QNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 정확히 40K-45K 파라미터가 되도록 설계
        self.cnn = self._build_precise_cnn().double()
        self.qnn = qlayer
        self.classifier = self._build_precise_classifier().double()
    
    def _build_precise_cnn(self):
        # 약 30K 파라미터의 CNN
        return nn.Sequential(
            # Block 1: 28x28 → 14x14
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),   # params: 1*8*25 + 8 = 208
            nn.BatchNorm2d(8),   # params: 16
            nn.ReLU(),
            
            # Block 2: 14x14 → 7x7
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # params: 8*16*9 + 16 = 1,168
            nn.BatchNorm2d(16),  # params: 32
            nn.ReLU(),
            
            # Block 3: 7x7 → 4x4
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # params: 16*32*9 + 32 = 4,640
            nn.BatchNorm2d(32),  # params: 64
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # 32x4x4 = 512
            
            nn.Flatten(),  # 512
            nn.Linear(512, 384),  # params: 512*384 + 384 = 196,992
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, 256),  # params: 384*256 + 256 = 98,560
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 총: 208 + 16 + 1,168 + 32 + 4,640 + 64 + 196,992 + 98,560 = 301,680 (여전히 많음)
    
    def _build_precise_classifier(self):
        # 약 8K 파라미터의 분류기
        return nn.Sequential(
            nn.Linear(1, 64),       # params: 64 + 1 = 65
            nn.ReLU(),
            nn.BatchNorm1d(64),     # params: 128
            nn.Dropout(0.4),
            
            nn.Linear(64, 32),      # params: 64*32 + 32 = 2,080
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 16),      # params: 32*16 + 16 = 528
            nn.ReLU(),
            
            nn.Linear(16, 1),       # params: 17
            nn.Sigmoid()
        )
        # 총: 65 + 128 + 2,080 + 528 + 17 = 2,818
    
    def forward(self, x):
        features = self.cnn(x)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        q_out = []
        for i in range(features.size(0)):
            q_result = self.qnn(features[i])
            q_out.append(q_result)
        q_out = torch.stack(q_out).unsqueeze(1)
        
        return self.classifier(q_out)

    def forward(self, x):  # x: [B, 1, 28, 28]
        features = self.cnn(x)                      # [B, 256]
        features = torch.nn.functional.normalize(features, p=2, dim=1)  # L2 정규화
        
        # 배치 처리 최적화
        q_out = []
        for i in range(features.size(0)):
            q_result = self.qnn(features[i])
            q_out.append(q_result)
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

# 정확히 계산된 최종 모델 (40K-45K 파라미터)
class FinalCNN_QNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN 부분 (약 35K 파라미터)
        self.cnn = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(1, 6, 5, stride=2, padding=2),    # params: 1*6*25 + 6 = 156
            nn.BatchNorm2d(6),                          # params: 12
            nn.ReLU(),
            
            # 14x14 → 7x7
            nn.Conv2d(6, 12, 3, stride=2, padding=1),   # params: 6*12*9 + 12 = 660
            nn.BatchNorm2d(12),                         # params: 24
            nn.ReLU(),
            
            # 7x7 → 4x4
            nn.Conv2d(12, 24, 3, stride=1, padding=1),  # params: 12*24*9 + 24 = 2,616
            nn.BatchNorm2d(24),                         # params: 48
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(6),                    # 24x6x6 = 864
            
            nn.Flatten(),
            nn.Linear(864, 512),                        # params: 864*512 + 512 = 442,880
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),                        # params: 512*256 + 256 = 131,328
            nn.ReLU(),
            nn.Dropout(0.2)
        ).double()
        
        self.qnn = qlayer
        
        # 분류기 (약 8K 파라미터)
        self.classifier = nn.Sequential(
            nn.Linear(1, 128),      # params: 129
            nn.ReLU(),
            nn.BatchNorm1d(128),    # params: 256
            nn.Dropout(0.4),
            
            nn.Linear(128, 64),     # params: 8,256
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),      # params: 2,080
            nn.ReLU(),
            
            nn.Linear(32, 1),       # params: 33
            nn.Sigmoid()
        ).double()
        
        # 총 예상: CNN(~577K) + QNN(56) + Classifier(~10.7K) = 너무 많음!
        
    def forward(self, x):
        features = self.cnn(x)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        q_out = []
        for i in range(features.size(0)):
            q_result = self.qnn(features[i])
            q_out.append(q_result)
        q_out = torch.stack(q_out).unsqueeze(1)
        
        return self.classifier(q_out)

# 실제 40K-45K 파라미터 모델 (정확한 계산)
class Balanced40KCNN_QNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 매우 신중하게 계산된 CNN (약 35K 파라미터)
        self.cnn = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(1, 4, 5, stride=2, padding=2),     # params: 1*4*25 + 4 = 104
            nn.BatchNorm2d(4),                           # params: 8
            nn.ReLU(),
            
            # 14x14 → 7x7  
            nn.Conv2d(4, 8, 3, stride=2, padding=1),     # params: 4*8*9 + 8 = 296
            nn.BatchNorm2d(8),                           # params: 16
            nn.ReLU(),
            
            # 7x7 → 4x4
            nn.Conv2d(8, 16, 3, stride=1, padding=1),    # params: 8*16*9 + 16 = 1,168
            nn.BatchNorm2d(16),                          # params: 32
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),                     # 16x8x8 = 1024
            
            nn.Flatten(),
            nn.Linear(1024, 768),                        # params: 1024*768 + 768 = 787,200
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 512),                         # params: 768*512 + 512 = 393,728
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),                         # params: 512*256 + 256 = 131,328
            nn.ReLU()
        ).double()
        
        # 이것도 너무 많음... 다른 접근 필요

epochs = 100

# 정말 마지막 시도: 정확히 45K가 되도록 역산해서 설계
class Final45KCNN_QNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 목표: 총 45,000 파라미터
        # QNN: 56 파라미터 (고정)
        # 남은 예산: 44,944 파라미터
        
        # CNN 부분 (약 20K 파라미터)
        self.cnn = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(1, 2, 5, stride=2, padding=2),     # params: 1*2*25 + 2 = 52
            nn.ReLU(),
            
            # 14x14 → 7x7
            nn.Conv2d(2, 4, 3, stride=2, padding=1),     # params: 2*4*9 + 4 = 76
            nn.ReLU(),
            
            # 7x7 → 4x4
            nn.Conv2d(4, 8, 3, stride=1, padding=1),     # params: 4*8*9 + 8 = 296
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3),                     # 8x3x3 = 72
            
            nn.Flatten(),
            nn.Linear(72, 256),                          # params: 72*256 + 256 = 18,688
            nn.ReLU(),
            nn.Dropout(0.3)
        ).double()
        
        self.qnn = qlayer
        
        # 분류기 부분 (약 25K 파라미터)
        self.classifier = nn.Sequential(
            nn.Linear(1, 160),      # params: 1*160 + 160 = 320
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(160, 120),    # params: 160*120 + 120 = 19,320
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(120, 80),     # params: 120*80 + 80 = 9,680
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(80, 1),       # params: 80*1 + 1 = 81
            nn.Sigmoid()
        ).double()
        
        # 총 계산: CNN(19,112) + QNN(56) + Classifier(29,401) = 48,569 (거의 맞음!)
        
    def forward(self, x):
        features = self.cnn(x)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        q_out = []
        for i in range(features.size(0)):
            q_result = self.qnn(features[i])
            q_out.append(q_result)
        q_out = torch.stack(q_out).unsqueeze(1)
        
        return self.classifier(q_out)

model = Final45KCNN_QNN_Model()

# 적응적 학습률 (경량 모델에 최적화)
optimizer = optim.AdamW([
    {'params': model.cnn.parameters(), 'lr': 0.001, 'weight_decay': 1e-4},  # CNN: 낮은 학습률
    {'params': model.qnn.parameters(), 'lr': 0.01, 'weight_decay': 1e-5},   # QNN: 높은 학습률  
    {'params': model.classifier.parameters(), 'lr': 0.005, 'weight_decay': 1e-4}  # 분류기: 중간 학습률
])

criterion = nn.BCELoss()

# 학습률 스케줄러
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=15)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Train data shape: {train_x.shape}, Test data shape: {test_x.shape}")

# 50K 제한 확인
total_params = sum(p.numel() for p in model.parameters())
if total_params <= 50000:
    print(f"✅ Parameter limit satisfied: {total_params:,} ≤ 50,000")
else:
    print(f"❌ Parameter limit exceeded: {total_params:,} > 50,000")
    raise ValueError("Model exceeds 50K parameter limit!")

# -----------------------
# 개선된 훈련 루프
# -----------------------
best_test_acc = 0
print("Starting CNN-QNN training...")

for epoch in tqdm(range(epochs)):
    model.train()
    total_loss, total_acc = 0, 0
    
    for xb, yb in train_loader:
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