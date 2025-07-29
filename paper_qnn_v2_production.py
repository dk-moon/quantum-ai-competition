import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import math
import time
import json
import os
from typing import Dict, List, Optional

# -----------------------
# 실제 테스트용 Production 버전
# -----------------------

class ProductionPaperQNN(nn.Module):
    """
    실제 테스트용 Paper QNN
    - 성능 최적화
    - 메모리 효율성
    - 확장성
    - 로깅 및 모니터링
    """
    
    def __init__(self, 
                 ensemble_type: str = "general",
                 n_qubits: int = 8,
                 frqi_iterations: int = 50,  # 실제 테스트용으로 축소
                 cache_limit: int = 1000):   # 메모리 관리
        super().__init__()
        
        self.ensemble_type = ensemble_type
        self.n_qubits = n_qubits
        self.frqi_iterations = frqi_iterations
        self.cache_limit = cache_limit
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # FRQI 캐시 (메모리 제한)
        self.frqi_cache = {}
        self.cache_access_count = {}
        
        # 성능 모니터링
        self.timing_stats = {
            'frqi_encoding': [],
            'quantum_forward': [],
            'total_forward': []
        }
        
        # 양자 텐서 트레인 분류기 매개변수 (증가)
        if ensemble_type == "general":
            params_per_gate = 14  # 실제 사용하는 매개변수 수
        else:
            params_per_gate = 6
        
        n_classifier_params = 3 * (self.n_qubits - 1) * params_per_gate + 20
        
        self.classifier_params = nn.Parameter(
            torch.randn(n_classifier_params, dtype=torch.float64) * 0.1
        )
        
        # 강화된 고전 후처리 네트워크
        self.classical_head = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        ).double()
        
        print(f"🚀 Production QNN initialized:")
        print(f"  - Ensemble: {ensemble_type}")
        print(f"  - Qubits: {self.n_qubits}")
        print(f"  - Classifier params: {n_classifier_params}")
        print(f"  - FRQI iterations: {frqi_iterations}")
        print(f"  - Cache limit: {cache_limit}")
    
    def _manage_cache(self):
        """메모리 효율적인 캐시 관리"""
        if len(self.frqi_cache) > self.cache_limit:
            # LRU 방식으로 캐시 정리
            sorted_items = sorted(
                self.cache_access_count.items(), 
                key=lambda x: x[1]
            )
            
            # 가장 적게 사용된 항목들 제거
            items_to_remove = len(self.frqi_cache) - self.cache_limit + 100
            for key, _ in sorted_items[:items_to_remove]:
                if key in self.frqi_cache:
                    del self.frqi_cache[key]
                    del self.cache_access_count[key]
    
    def preprocess_image_to_frqi(self, image: torch.Tensor) -> torch.Tensor:
        """최적화된 FRQI 전처리"""
        # 28x28을 32x32로 패딩 (논문과 동일)
        if image.shape[-1] == 28:
            pad = (2, 2, 2, 2)
            image = torch.nn.functional.pad(image, pad, mode='constant', value=0)
        
        # 효율적인 정규화
        flat_image = image.flatten()
        img_min, img_max = flat_image.min(), flat_image.max()
        normalized = (flat_image - img_min) / (img_max - img_min + 1e-8)
        
        # FRQI 각도 변환
        angles = normalized * math.pi / 2
        return angles[:self.n_qubits]
    
    def optimize_frqi_encoding_fast(self, image: torch.Tensor, image_id: str) -> torch.Tensor:
        """
        개선된 FRQI 인코딩 (더 나은 이미지 표현)
        """
        start_time = time.time()
        
        # 캐시 확인
        if image_id in self.frqi_cache:
            self.cache_access_count[image_id] = self.cache_access_count.get(image_id, 0) + 1
            return self.frqi_cache[image_id]
        
        # 더 나은 이미지 특징 추출
        target_angles = self.preprocess_image_to_frqi(image)
        
        # 이미지의 통계적 특징 활용
        img_mean = torch.mean(target_angles)
        img_std = torch.std(target_angles)
        img_max = torch.max(target_angles)
        img_min = torch.min(target_angles)
        
        # 간단하고 빠른 인코딩 (학습 가능하도록)
        n_encoding_params = self.n_qubits
        
        # 이미지의 주요 특징을 직접 매핑
        encoding_params = torch.zeros(n_encoding_params, dtype=torch.float64)
        
        # 이미지를 8개 영역으로 나누어 각 영역의 평균값 사용
        img_2d = image.squeeze().reshape(28, 28) if image.dim() > 2 else image.reshape(28, 28)
        
        # 8개 영역의 평균값을 각도로 변환
        regions = [
            img_2d[:14, :14].mean(),    # 좌상
            img_2d[:14, 14:].mean(),    # 우상  
            img_2d[14:, :14].mean(),    # 좌하
            img_2d[14:, 14:].mean(),    # 우하
            img_2d[:, :14].mean(),      # 좌측
            img_2d[:, 14:].mean(),      # 우측
            img_2d[:14, :].mean(),      # 상단
            img_2d[14:, :].mean()       # 하단
        ]
        
        for i in range(n_encoding_params):
            # 정규화된 영역 평균을 각도로 변환
            region_val = regions[i] if i < len(regions) else img_mean
            encoding_params[i] = (region_val + 1) * math.pi / 4  # [-1,1] -> [0,π/2]
        
        # 유효한 각도 범위로 클램핑
        encoding_params = torch.clamp(encoding_params, 0, math.pi/2)
        
        # 캐시 저장 및 관리
        self.frqi_cache[image_id] = encoding_params
        self.cache_access_count[image_id] = 1
        self._manage_cache()
        
        # 성능 기록
        encoding_time = time.time() - start_time
        self.timing_stats['frqi_encoding'].append(encoding_time)
        
        return self.frqi_cache[image_id]
    
    def quantum_tensor_train_classifier_optimized(self, frqi_params, classifier_params):
        """최적화된 양자 텐서 트레인 분류기"""
        
        @qml.qnode(self.dev, interface="torch")
        def circuit():
            # 1. FRQI 인코딩된 상태 준비
            for i in range(min(self.n_qubits, len(frqi_params))):
                qml.RY(frqi_params[i], wires=i)
            
            # 2. 강화된 양자 텐서 트레인 레이어들
            param_idx = 0
            n_layers = 2  # 빠른 학습을 위해 축소
            
            for layer in range(n_layers):
                for i in range(self.n_qubits - 1):
                    if self.ensemble_type == "general":
                        if param_idx + 15 <= len(classifier_params):
                            gate_params = classifier_params[param_idx:param_idx + 15]
                            
                            # 효율적인 SU(4) 구현
                            qml.RY(gate_params[0], wires=i)
                            qml.RZ(gate_params[1], wires=i)
                            qml.RY(gate_params[2], wires=i+1)
                            qml.RZ(gate_params[3], wires=i+1)
                            
                            qml.CNOT(wires=[i, i+1])
                            
                            qml.RY(gate_params[4], wires=i)
                            qml.RZ(gate_params[5], wires=i)
                            qml.RY(gate_params[6], wires=i+1)
                            qml.RZ(gate_params[7], wires=i+1)
                            
                            qml.CNOT(wires=[i+1, i])
                            
                            qml.RY(gate_params[8], wires=i)
                            qml.RZ(gate_params[9], wires=i)
                            qml.RY(gate_params[10], wires=i+1)
                            qml.RZ(gate_params[11], wires=i+1)
                            
                            # 추가 얽힘
                            qml.CNOT(wires=[i, i+1])
                            
                            qml.RY(gate_params[12], wires=i)
                            qml.RY(gate_params[13], wires=i+1)
                            
                            param_idx += 14  # 더 많은 매개변수 사용
                    
                    else:  # sparse ensemble
                        if param_idx + 6 <= len(classifier_params):
                            gate_params = classifier_params[param_idx:param_idx + 6]
                            
                            qml.RY(gate_params[0], wires=i)
                            qml.RZ(gate_params[1], wires=i)
                            qml.RY(gate_params[2], wires=i+1)
                            qml.RZ(gate_params[3], wires=i+1)
                            
                            # 조건부 CNOT
                            if abs(gate_params[4].item()) > 0.1:
                                qml.CNOT(wires=[i, i+1])
                            
                            qml.RY(gate_params[5], wires=i)
                            param_idx += 6
            
            # 3. 판독층
            if param_idx < len(classifier_params):
                qml.RY(classifier_params[param_idx], wires=0)
            
            return qml.expval(qml.PauliZ(0))
        
        return circuit()
    
    def forward(self, x: torch.Tensor, image_ids: Optional[List[str]] = None) -> torch.Tensor:
        """최적화된 순전파"""
        start_time = time.time()
        batch_size = x.size(0)
        results = []
        
        for i in range(batch_size):
            image = x[i]
            image_id = image_ids[i] if image_ids is not None else f"batch_{i}_{time.time()}"
            
            # 1. 빠른 FRQI 인코딩
            frqi_start = time.time()
            frqi_params = self.optimize_frqi_encoding_fast(image, image_id)
            frqi_time = time.time() - frqi_start
            
            # 2. 양자 분류
            quantum_start = time.time()
            quantum_output = self.quantum_tensor_train_classifier_optimized(
                frqi_params, self.classifier_params
            )
            quantum_time = time.time() - quantum_start
            
            results.append(quantum_output)
            
            # 성능 기록
            self.timing_stats['quantum_forward'].append(quantum_time)
        
        # 배치 결과 결합
        quantum_batch = torch.stack(results).unsqueeze(1)
        
        # 고전 후처리
        final_output = self.classical_head(quantum_batch)
        
        # 전체 시간 기록
        total_time = time.time() - start_time
        self.timing_stats['total_forward'].append(total_time)
        
        return final_output
    
    def get_performance_stats(self) -> Dict:
        """성능 통계 반환"""
        stats = {}
        for key, times in self.timing_stats.items():
            if times:
                stats[key] = {
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }
        
        stats['cache_info'] = {
            'size': len(self.frqi_cache),
            'limit': self.cache_limit,
            'hit_rate': len([c for c in self.cache_access_count.values() if c > 1]) / max(1, len(self.cache_access_count))
        }
        
        return stats
    
    def save_model(self, path: str):
        """모델 저장 (캐시 포함)"""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'frqi_cache': self.frqi_cache,
            'cache_access_count': self.cache_access_count,
            'config': {
                'ensemble_type': self.ensemble_type,
                'n_qubits': self.n_qubits,
                'frqi_iterations': self.frqi_iterations,
                'cache_limit': self.cache_limit
            },
            'performance_stats': self.get_performance_stats()
        }
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """모델 로드 (캐시 포함)"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.frqi_cache = checkpoint.get('frqi_cache', {})
        self.cache_access_count = checkpoint.get('cache_access_count', {})
        print(f"Model loaded from {path}")
        print(f"Cache entries: {len(self.frqi_cache)}")


# -----------------------
# 실제 테스트용 데이터 로더
# -----------------------
def load_full_fashion_mnist():
    """전체 Fashion-MNIST 데이터 로드"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    
    train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    def filter_binary(dataset):
        indices = [i for i, (_, y) in enumerate(dataset) if y in [0, 6]]  # T-shirt vs Shirt
        data = torch.stack([dataset[i][0] for i in indices])
        labels = torch.tensor([int(dataset[i][1] == 6) for i in indices])
        return data, labels

    return filter_binary(train_set), filter_binary(test_set)


def create_production_dataloaders(train_size: int = 5000, test_size: int = 1000, batch_size: int = 16):
    """실제 테스트용 데이터로더 생성"""
    (train_x, train_y), (test_x, test_y) = load_full_fashion_mnist()
    
    # 데이터 타입 변환
    train_x = train_x.double()[:train_size]
    test_x = test_x.double()[:test_size]
    train_y = train_y.double().unsqueeze(1)[:train_size]
    test_y = test_y.double().unsqueeze(1)[:test_size]
    
    # 데이터로더 생성
    train_loader = DataLoader(
        TensorDataset(train_x, train_y), 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # 양자 회로와 호환성을 위해
    )
    
    test_loader = DataLoader(
        TensorDataset(test_x, test_y), 
        batch_size=batch_size,
        num_workers=0
    )
    
    return train_loader, test_loader


# -----------------------
# 실제 테스트용 훈련 함수
# -----------------------
def train_production_model(model, train_loader, test_loader, epochs: int = 50):
    """실제 테스트용 훈련 함수"""
    
    # 개선된 최적화 설정 (더 빠른 학습)
    optimizer = optim.Adam([
        {'params': model.classifier_params, 'lr': 0.01, 'weight_decay': 1e-4},  # 학습률 10배 증가
        {'params': model.classical_head.parameters(), 'lr': 0.005, 'weight_decay': 1e-3}  # 학습률 5배 증가
    ])
    
    # 클래스 불균형 고려한 손실 함수
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # 더 적극적인 스케줄링
    
    # 훈련 기록
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_test_acc = 0
    patience_counter = 0
    patience = 35
    
    print(f"\n🚀 Starting Production Training:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # 훈련
        model.train()
        total_loss, total_acc = 0, 0
        
        for batch_idx, (xb, yb) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
            optimizer.zero_grad()
            
            # 이미지 해시 기반 ID (캐시 효율성)
            image_ids = []
            for i in range(xb.size(0)):
                img_hash = hash(xb[i].flatten().sum().item())
                image_ids.append(f"img_{img_hash}")
            
            preds = model(xb, image_ids)
            loss = criterion(preds, yb)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            acc = ((torch.sigmoid(preds) > 0.5).double() == yb).double().mean()
            total_loss += loss.item()
            total_acc += acc.item()
        
        # 평가
        model.eval()
        test_loss, test_acc = 0, 0
        with torch.no_grad():
            for batch_idx, (xb, yb) in enumerate(test_loader):
                image_ids = []
                for i in range(xb.size(0)):
                    img_hash = hash(xb[i].flatten().sum().item())
                    image_ids.append(f"img_{img_hash}")
                preds = model(xb, image_ids)
                loss = criterion(preds, yb)
                acc = ((torch.sigmoid(preds) > 0.5).double() == yb).double().mean()
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
        
        scheduler.step()  # StepLR은 인자 불필요
        
        # 최고 성능 체크
        if avg_test_acc > best_test_acc:
            best_test_acc = avg_test_acc
            patience_counter = 0
            model.save_model('best_production_qnn.pth')
        else:
            patience_counter += 1
        
        # 로깅
        if epoch % 5 == 0 or patience_counter == 0:
            print(f"[Epoch {epoch+1}] Train Acc: {avg_train_acc:.4f} | Test Acc: {avg_test_acc:.4f} | Best: {best_test_acc:.4f}")
            
            # 성능 통계 출력
            stats = model.get_performance_stats()
            if 'total_forward' in stats:
                print(f"  Avg forward time: {stats['total_forward']['mean']:.3f}s")
                print(f"  Cache hit rate: {stats['cache_info']['hit_rate']:.2f}")
        
        # 조기 종료
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'best_test_acc': best_test_acc,
        'performance_stats': model.get_performance_stats()
    }


# -----------------------
# 메인 실행
# -----------------------
if __name__ == "__main__":
    print("🚀 Production Paper QNN - Real Test Implementation")
    print("=" * 70)
    
    # 전체 데이터 설정
    TRAIN_SIZE = 12000  # 전체 훈련 데이터 (클래스 0: 6000 + 클래스 6: 6000)
    TEST_SIZE = 2000    # 전체 테스트 데이터 (클래스 0: 1000 + 클래스 6: 1000)
    BATCH_SIZE = 32     # 더 큰 배치 크기
    EPOCHS = 300        # 전체 데이터이므로 에포크 수 축소
    
    print(f"📊 Test Configuration:")
    print(f"  - Train size: {TRAIN_SIZE}")
    print(f"  - Test size: {TEST_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Epochs: {EPOCHS}")
    
    # 데이터 로드
    train_loader, test_loader = create_production_dataloaders(
        train_size=TRAIN_SIZE, 
        test_size=TEST_SIZE, 
        batch_size=BATCH_SIZE
    )
     
    print(f"\n✅ Data loaded successfully")
    
    # 모델 초기화
    model = ProductionPaperQNN(
        ensemble_type="general",
        n_qubits=8,
        frqi_iterations=30,  # 실제 테스트용
        cache_limit=12000  # 전체 데이터에 맞게 증가
    )
    
    # 매개변수 수 계산
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📈 Model Info:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Parameter limit check: {total_params <= 50000} (≤50K)")
    
    # 실제 훈련 실행
    print(f"\n🏋️ Starting Real Training...")
    results = train_production_model(model, train_loader, test_loader, epochs=EPOCHS)
    
    # 최종 결과
    print(f"\n🎯 Final Results:")
    print(f"  - Best Test Accuracy: {results['best_test_acc']:.4f}")
    print(f"  - Total epochs: {len(results['train_accs'])}")
    
    # 성능 통계
    perf_stats = results['performance_stats']
    if 'total_forward' in perf_stats:
        print(f"  - Avg forward time: {perf_stats['total_forward']['mean']:.3f}s")
        print(f"  - Cache hit rate: {perf_stats['cache_info']['hit_rate']:.2f}")
        print(f"  - Cache size: {perf_stats['cache_info']['size']}")
    
    # 결과 저장
    with open('production_results.json', 'w') as f:
        json.dump({
            'best_accuracy': results['best_test_acc'],
            'final_epoch': len(results['train_accs']),
            'performance_stats': perf_stats,
            'config': {
                'train_size': TRAIN_SIZE,
                'test_size': TEST_SIZE,
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS
            }
        }, f, indent=2)
    
    print(f"\n✅ Production test completed!")
    print(f"Results saved to: production_results.json")
    print(f"Best model saved to: best_production_qnn.pth")