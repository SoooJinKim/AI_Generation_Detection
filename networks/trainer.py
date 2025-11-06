"""
Modularized Trainer
모델과 데이터로더를 인자로 받아서 학습을 수행하는 Trainer
Single-Tower와 Multi-Tower 모두 지원
"""

import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel
from networks.multi_tower import MultiTowerModel, SingleTowerWrapper


class Trainer:
    """
    모듈화된 Trainer 클래스
    모델과 데이터로더를 인자로 받아서 학습 수행
    """
    def __init__(self, model, device='cuda', lr=1e-4, optim='adam', beta1=0.9):
        """
        Args:
            model: 학습할 모델 (nn.Module)
            device: 디바이스 ('cuda' or 'cpu')
            lr: learning rate
            optim: optimizer 타입 ('adam' or 'sgd')
            beta1: Adam의 beta1 파라미터
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Optimizer 초기화
        if optim == 'adam':
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr, 
                betas=(beta1, 0.999)
            )
        elif optim == 'sgd':
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr, 
                momentum=0.0, 
                weight_decay=0
            )
        else:
            raise ValueError(f"optim should be 'adam' or 'sgd', got: {optim}")
        
        self.lr = lr
        self.total_steps = 0
        self.loss = 0.0
        
        # Multi-Tower 지원
        self.is_multi_tower = isinstance(self.model, MultiTowerModel) or \
                             (isinstance(self.model, nn.DataParallel) and 
                              isinstance(self.model.module, MultiTowerModel))
    
    def train(self):
        """학습 모드로 전환"""
        self.model.train()
    
    def eval(self):
        """평가 모드로 전환"""
        self.model.eval()
    
    def set_input(self, batch):
        """
        배치 데이터 설정
        
        Args:
            batch: (images, labels) 또는 (images_list, labels) for Multi-Tower
                   images: single tensor or list of tensors
                   labels: tensor
        """
        images, labels = batch
        
        # Multi-Tower: images가 리스트인 경우
        if isinstance(images, list):
            self.input = [img.to(self.device) for img in images]
        else:
            self.input = images.to(self.device)
        
        self.label = labels.to(self.device).float()
    
    def forward(self):
        """Forward pass"""
        self.output = self.model(self.input)
        return self.output
    
    def backward(self):
        """Backward pass"""
        self.loss.backward()
    
    def optimize_parameters(self):
        """한 스텝 학습 수행"""
        # Forward
        self.forward()
        
        # Loss 계산
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        
        # Backward
        self.optimizer.zero_grad()
        self.backward()
        
        # Update
        self.optimizer.step()
        
        return self.loss.item()
    
    def get_loss(self):
        """현재 loss 반환"""
        return self.loss
    
    def adjust_learning_rate(self, min_lr=1e-6, decay_rate=0.9):
        """Learning rate 조정"""
        old_lr = self.optimizer.param_groups[0]['lr']
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay_rate
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr
                print(f"⚠️ Learning rate reached minimum: {min_lr}")
                return False
        
        self.lr = self.optimizer.param_groups[0]['lr']
        print('*' * 50)
        print(f'🔧 Learning rate adjusted: {old_lr:.6f} -> {self.lr:.6f}')
        print('*' * 50)
        return True
    
    def save_checkpoint(self, path):
        """체크포인트 저장"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'lr': self.lr,
        }
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, path, load_optimizer=True):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'total_steps' in checkpoint:
            self.total_steps = checkpoint['total_steps']
        
        if 'lr' in checkpoint:
            self.lr = checkpoint['lr']
        
        print(f"✅ Checkpoint loaded: {path}")
    
    def get_current_learning_rate(self):
        """현재 learning rate 반환"""
        return self.optimizer.param_groups[0]['lr']


# Legacy Trainer (backward compatibility)
class LegacyTrainer(BaseModel):
    """
    기존 코드와의 호환성을 위한 Legacy Trainer
    """
    def name(self):
        return 'LegacyTrainer'

    def __init__(self, opt):
        super(LegacyTrainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(pretrained=False, num_classes=1)

        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])
 

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.9
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.9} to {param_group["lr"]}')
        print('*'*25)
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

