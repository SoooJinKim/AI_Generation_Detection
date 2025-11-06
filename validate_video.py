"""
동영상 검증 스크립트

동영상 파일에 대해 여러 프레임을 샘플링하여 예측
하나라도 fake로 예측되면 전체를 fake로 처리

Label: 0=Real, 1=Fake
"""

import torch
import numpy as np
import os
from pathlib import Path
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, accuracy_score
from options.test_options import TestOptions
from data.datasets import load_video_frames, TransformBuilder
from tqdm import tqdm


def predict_video(model, video_path, transform, device, num_frames=5, sampling='uniform'):
    """
    동영상에 대한 예측
    
    Args:
        model: PyTorch 모델
        video_path: 동영상 경로
        transform: Transformation pipeline
        device: 'cuda' or 'cpu'
        num_frames: 샘플링할 프레임 수
        sampling: 'uniform' 또는 'random'
        
    Returns:
        final_pred: 최종 예측 (0=Real, 1=Fake)
        max_prob: 최대 fake 확률
        frame_probs: 각 프레임별 fake 확률 리스트
    """
    try:
        # 여러 프레임 샘플링
        frames = load_video_frames(video_path, num_frames=num_frames, sampling=sampling)
        
        frame_probs = []
        with torch.no_grad():
            for frame in frames:
                # Transform 적용
                transformed = transform(frame).unsqueeze(0).to(device)
                
                # 예측
                output = model(transformed).sigmoid().item()
                frame_probs.append(output)
        
        # 하나라도 fake(threshold > 0.5)면 fake로 처리
        max_prob = max(frame_probs)
        final_pred = 1 if max_prob > 0.5 else 0
        
        return final_pred, max_prob, frame_probs
        
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None, None, None


def validate_videos(model, opt, num_frames=5, sampling='uniform'):
    """
    동영상 데이터셋 검증
    
    Args:
        model: PyTorch 모델
        opt: TestOptions
        num_frames: 동영상당 샘플링할 프레임 수
        sampling: 'uniform' 또는 'random'
    """
    device = next(model.parameters()).device
    
    # Transform 생성
    transform = TransformBuilder.build_transform(
        getattr(opt, 'transform_mode', 'texture'),
        opt,
        is_train=False
    )
    
    # 데이터 경로
    data_root = Path(opt.dataroot)
    
    # Real과 Fake 디렉토리
    real_dir = data_root / 'real'
    fake_dir = data_root / 'fake'
    
    # 동영상 파일 수집
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    
    real_videos = [f for f in real_dir.glob('*') if f.suffix.lower() in VIDEO_EXTENSIONS]
    fake_videos = [f for f in fake_dir.glob('*') if f.suffix.lower() in VIDEO_EXTENSIONS]
    
    print(f"📹 Found {len(real_videos)} real videos, {len(fake_videos)} fake videos")
    print(f"🔍 Sampling {num_frames} frames per video ({sampling} sampling)")
    
    y_true, y_pred = [], []
    
    # Real 동영상 처리
    print("\n처리 중: Real videos...")
    for video_path in tqdm(real_videos):
        pred, max_prob, frame_probs = predict_video(
            model, str(video_path), transform, device, num_frames, sampling
        )
        if pred is not None:
            y_true.append(0)  # Real
            y_pred.append(max_prob)
    
    # Fake 동영상 처리
    print("\n처리 중: Fake videos...")
    for video_path in tqdm(fake_videos):
        pred, max_prob, frame_probs = predict_video(
            model, str(video_path), transform, device, num_frames, sampling
        )
        if pred is not None:
            y_true.append(1)  # Fake
            y_pred.append(max_prob)
    
    # 결과 계산
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 전체 정확도
    acc = accuracy_score(y_true, y_pred > 0.5)
    
    # Real 정확도 (label=0)
    if (y_true == 0).any():
        r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    else:
        r_acc = 0.0
    
    # Fake 정확도 (label=1)
    if (y_true == 1).any():
        f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    else:
        f_acc = 0.0
    
    # Average Precision
    if len(np.unique(y_true)) > 1:
        ap = average_precision_score(y_true, y_pred)
    else:
        ap = 0.0
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 Video Validation Results")
    print("=" * 60)
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Real Accuracy:    {r_acc:.4f}")
    print(f"Fake Accuracy:    {f_acc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print("=" * 60)
    
    return acc, ap, r_acc, f_acc, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    # 모델 로드
    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    
    # GPU/CPU 선택
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"💻 Using device: {device}")

    # 동영상 검증
    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate_videos(
        model, opt, num_frames=5, sampling='uniform'
    )

