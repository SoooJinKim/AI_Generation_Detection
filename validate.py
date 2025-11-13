import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader
from tqdm import tqdm

def validate(model, opt):
    """
    모델 검증
    
    Note: label=0은 Real, label=1은 Fake
    동영상의 경우: 여러 프레임 샘플링 → 각 프레임 예측 → voting
    """
    data_loader = create_dataloader(opt)
    
    # GPU/CPU 자동 선택
    device = next(model.parameters()).device
    
    # Voting 방식
    voting_method = getattr(opt, 'video_voting', 'max')

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating", total=len(data_loader)):
            # 배치 언패킹
            if len(batch) == 4:
                data, labels, paths, is_video_flags = batch
            else:
                # Legacy format
                if len(batch) == 3:
                    data, labels, paths = batch
                else:
                    data, labels = batch
                is_video_flags = None
            
            # 동영상 처리
            if is_video_flags and any(is_video_flags):
                for sample_idx, is_video in enumerate(is_video_flags):
                    label = labels[sample_idx].item()
                    
                    if is_video:
                        # 동영상: 여러 프레임 예측 → voting
                        frames = data[sample_idx]  # List[Tensor]
                        frame_preds = []
                        
                        for frame in frames:
                            frame_batch = frame.unsqueeze(0).to(device)
                            pred = model(frame_batch).sigmoid().item()
                            frame_preds.append(pred)
                        
                        # Voting
                        if voting_method == 'max':
                            final_pred = max(frame_preds)
                        elif voting_method == 'avg':
                            final_pred = np.mean(frame_preds)
                        elif voting_method == 'majority':
                            # 과반수 투표
                            votes = [1 if p > 0.5 else 0 for p in frame_preds]
                            final_pred = 1.0 if sum(votes) > len(votes) / 2 else 0.0
                        else:
                            final_pred = max(frame_preds)
                        
                        y_pred.append(final_pred)
                        y_true.append(label)
                    else:
                        # 이미지: 일반 처리
                        img = data[sample_idx].unsqueeze(0).to(device)
                        pred = model(img).sigmoid().item()
                        y_pred.append(pred)
                        y_true.append(label)
            else:
                # 일반 이미지 배치
                in_tens = data.to(device)
                preds = model(in_tens).sigmoid().flatten().tolist()
                y_pred.extend(preds)
                y_true.extend(labels.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Real(0)과 Fake(1) 정확도 계산
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
