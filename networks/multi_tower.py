"""
Multi-Tower 아키텍처 구현
여러 transformation을 통과한 이미지들을 각각의 tower에서 처리하고
결과를 통합하여 최종 예측을 생성
"""

import torch
import torch.nn as nn
from networks.resnet import resnet50


class MultiTowerModel(nn.Module):
    """
    Multi-Tower 아키텍처
    
    입력 -> 3개의 transformation -> 각 tower에서 처리 -> 통합 -> 단일 출력
    """
    def __init__(self, num_towers=3, num_classes=1, pretrained=False, fusion_method='concat'):
        """
        Args:
            num_towers: tower 개수 (transformation 개수와 동일)
            num_classes: 출력 클래스 수
            pretrained: 사전학습 모델 사용 여부
            fusion_method: 'concat', 'average', 'attention'
        """
        super(MultiTowerModel, self).__init__()
        
        self.num_towers = num_towers
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        
        # 각 tower는 독립적인 ResNet50 backbone
        self.towers = nn.ModuleList([
            resnet50(pretrained=pretrained, num_classes=512)  # 512 feature dim
            for _ in range(num_towers)
        ])
        
        # Fusion layer
        if fusion_method == 'concat':
            # 모든 tower의 feature를 concatenate
            self.fusion = nn.Sequential(
                nn.Linear(512 * num_towers, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        elif fusion_method == 'average':
            # 평균을 낸 후 분류
            self.fusion = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        elif fusion_method == 'attention':
            # Attention mechanism을 사용한 weighted fusion
            self.attention = nn.Sequential(
                nn.Linear(512, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
            self.fusion = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")
    
    def forward(self, x):
        """
        Args:
            x: List of tensors [tower1_input, tower2_input, tower3_input]
               OR single tensor (batch_size, C, H, W) for single tower mode
               
        Returns:
            output: (batch_size, num_classes)
        """
        # Single input인 경우 (backward compatibility)
        if not isinstance(x, (list, tuple)):
            x = [x]
        
        # 각 tower를 통과시켜 feature 추출
        tower_features = []
        for i, tower_input in enumerate(x):
            if i < len(self.towers):
                feat = self.towers[i](tower_input)
                tower_features.append(feat)
        
        # Fusion
        if self.fusion_method == 'concat':
            # [B, 512*num_towers]
            combined = torch.cat(tower_features, dim=1)
            output = self.fusion(combined)
            
        elif self.fusion_method == 'average':
            # [B, 512]
            combined = torch.stack(tower_features, dim=0).mean(dim=0)
            output = self.fusion(combined)
            
        elif self.fusion_method == 'attention':
            # Attention weights: [num_towers, B, 1]
            attention_weights = []
            for feat in tower_features:
                att = self.attention(feat)  # [B, 1]
                attention_weights.append(att)
            
            attention_weights = torch.stack(attention_weights, dim=0)  # [num_towers, B, 1]
            attention_weights = torch.softmax(attention_weights, dim=0)  # Normalize across towers
            
            # Weighted sum: [B, 512]
            tower_features_stacked = torch.stack(tower_features, dim=0)  # [num_towers, B, 512]
            combined = (tower_features_stacked * attention_weights).sum(dim=0)  # [B, 512]
            output = self.fusion(combined)
        
        return output
    
    def get_tower(self, idx):
        """특정 tower 모델 반환"""
        return self.towers[idx]
    
    def freeze_towers(self, tower_indices=None):
        """특정 tower들을 freeze"""
        if tower_indices is None:
            tower_indices = range(self.num_towers)
        
        for idx in tower_indices:
            for param in self.towers[idx].parameters():
                param.requires_grad = False
    
    def unfreeze_towers(self, tower_indices=None):
        """특정 tower들을 unfreeze"""
        if tower_indices is None:
            tower_indices = range(self.num_towers)
        
        for idx in tower_indices:
            for param in self.towers[idx].parameters():
                param.requires_grad = True


class SingleTowerWrapper(nn.Module):
    """
    Single Tower를 Multi-Tower 인터페이스로 감싸는 래퍼
    기존 코드와의 호환성 유지
    """
    def __init__(self, model):
        super(SingleTowerWrapper, self).__init__()
        self.model = model
        self.num_towers = 1
    
    def forward(self, x):
        # List 입력을 첫 번째 요소만 사용
        if isinstance(x, (list, tuple)):
            x = x[0]
        return self.model(x)
    
    def get_tower(self, idx):
        if idx == 0:
            return self.model
        raise IndexError(f"SingleTowerWrapper only has 1 tower, got idx={idx}")

