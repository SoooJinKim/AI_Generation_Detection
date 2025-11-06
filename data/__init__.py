import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from .datasets import dataset_folder
import os


def get_dataset(opt):
    """
    데이터셋 생성
    
    - realfake 모드: dataset/{real, fake} 구조를 자동으로 처리
    - binary 모드: 기존 방식 유지
    """
    # realfake 모드인 경우 (새로운 dataset/{real,fake} 구조)
    if opt.mode == 'realfake':
        return dataset_folder(opt, opt.dataroot)
    
    # binary 모드인 경우 (기존 방식)
    classes = os.listdir(opt.dataroot) if len(opt.classes) == 0 else opt.classes
    
    # 0_real, 0_fake 구조인 경우
    if '0_real' not in classes or '0_fake' not in classes:
        dset_lst = []
        for cls in classes:
            root = opt.dataroot + '/' + cls
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
        return torch.utils.data.ConcatDataset(dset_lst)
    
    return dataset_folder(opt, opt.dataroot)

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def video_aware_collate_fn(batch):
    """
    동영상과 이미지를 함께 처리하는 collate function
    
    Returns:
        - images or frames: 배치 텐서 또는 프레임 리스트
        - labels: 레이블 텐서
        - paths: 경로 리스트  
        - is_video_batch: 동영상 여부 리스트
    """
    # 첫 번째 샘플 확인
    first_sample = batch[0]
    
    # 4개 요소인지 확인 (image/frames, label, path, is_video)
    if len(first_sample) == 4:
        images_or_frames = []
        labels = []
        paths = []
        is_video_flags = []
        
        for sample in batch:
            data, label, path, is_video = sample
            
            if is_video:
                # 동영상: 프레임 리스트
                images_or_frames.append(data)  # List[Tensor]
            else:
                # 이미지: 단일 텐서
                images_or_frames.append(data)  # Tensor
            
            labels.append(label)
            paths.append(path)
            is_video_flags.append(is_video)
        
        # 동영상이 하나라도 있으면 특별 처리
        if any(is_video_flags):
            # 동영상 배치는 리스트로 반환 (각 샘플이 프레임 리스트)
            return images_or_frames, torch.tensor(labels), paths, is_video_flags
        else:
            # 모두 이미지면 일반 배치
            return torch.stack(images_or_frames), torch.tensor(labels), paths, is_video_flags
    
    # Legacy format (3개 요소)
    else:
        if isinstance(first_sample[0], list):
            # Multi-Tower
            num_transforms = len(first_sample[0])
            batched_transforms = []
            for i in range(num_transforms):
                transform_batch = torch.stack([sample[0][i] for sample in batch])
                batched_transforms.append(transform_batch)
            labels = torch.tensor([sample[1] for sample in batch])
            paths = [sample[2] for sample in batch]
            return batched_transforms, labels, paths
        else:
            # Single-Tower
            images = torch.stack([sample[0] for sample in batch])
            labels = torch.tensor([sample[1] for sample in batch])
            paths = [sample[2] for sample in batch]
            return images, labels, paths


def create_dataloader(opt):
    """
    데이터로더 생성
    Multi-Tower를 지원하는 collate_fn 적용
    """
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None
    
    # 항상 video_aware_collate_fn 사용 (동영상 지원)
    collate_fn = video_aware_collate_fn

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(opt.num_threads),
        collate_fn=collate_fn
    )
    return data_loader
