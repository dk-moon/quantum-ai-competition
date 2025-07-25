import torch
import torch.nn as nn

# 간단한 파라미터 계산
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 2, 5, stride=2, padding=2),  # 1*2*25 + 2 = 52
            nn.ReLU(),
            nn.Conv2d(2, 2, 3, stride=2, padding=1),  # 2*2*9 + 2 = 38
            nn.ReLU(),
            nn.Conv2d(2, 2, 3, stride=1, padding=1),  # 2*2*9 + 2 = 38
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(8, 8),  # 8*8 + 8 = 72
            nn.ReLU(),
            nn.Dropout(0.1)
        ).double()
        
        # QNN 파라미터 (7개)
        self.qnn_params = nn.Parameter(torch.randn(7, dtype=torch.float64) * 0.1)
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(1, 256),      # 1*256 + 256 = 512
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),    # 256*128 + 128 = 32,896
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),     # 128*64 + 64 = 8,256
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),      # 64*16 + 16 = 1,040
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),       # 16*1 + 1 = 17
        ).double()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = SimpleCNN()
total_params = count_parameters(model)
cnn_params = sum(p.numel() for p in model.cnn.parameters())
qnn_params = model.qnn_params.numel()
classifier_params = sum(p.numel() for p in model.classifier.parameters())

print(f'Total parameters: {total_params:,}')
print(f'CNN parameters: {cnn_params:,}')
print(f'QNN parameters: {qnn_params:,}')
print(f'Classifier parameters: {classifier_params:,}')
print(f'Parameter limit check: {total_params <= 45000} (≤45K)')
print(f'Sum check: {cnn_params + qnn_params + classifier_params} = {total_params}')