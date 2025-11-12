import os
import sys
import torch
import torch.nn as nn
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2 # OpenCV (동영상 처리에 필요)
from collections import OrderedDict
import torchvision.transforms as T

# --- 프로젝트의 기존 모듈 임포트 ---
# (train.py와 동일한 위치에 있어야 함)
try:
    from networks.resnet import resnet50
    from networks.multi_tower import SingleTowerWrapper
except ImportError:
    print("오류: 'networks' 폴더에서 resnet50 또는 SingleTowerWrapper를 임포트할 수 없습니다.")
    print("train.py와 동일한 폴더에서 실행해야 합니다.")
    sys.exit(1)

# ==========================================================
# 헬퍼 함수 1: 모델 로드
# ==========================================================
def load_model(model_path, device):
    """
    SingleTowerWrapper 모델을 생성하고 가중치를 로드합니다.
    """
    try:
        # 1. 모델 구조 생성 (train.py의 create_model 함수 로직과 동일)
        base_model = resnet50(pretrained=False, num_classes=1)
        model = SingleTowerWrapper(base_model)
        
        # 2. 저장된 가중치 불러오기
        state_dict = torch.load(model_path, map_location=device)

        # 3. DataParallel('module.') 접두사 제거 (필요시)
        cleaned_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            cleaned_state_dict[name] = v
            
        model.load_state_dict(cleaned_state_dict)
        
        model.to(device)
        model.eval() # 평가 모드로 설정
        print(f"✅ 모델 로드 완료: {model_path}")
        return model
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {model_path}")
        print(f"   오류: {e}")
        sys.exit(1)

# ==========================================================
# 헬퍼 함수 2: 전처리(Transform) 정의
# ==========================================================
def get_transform(mode='texture'):
    """
    ⚠️ [수정 필요!] ⚠️
    이 함수를 훈련(train.py) 시 사용했던 전처리 로직으로 교체해야 합니다.
    (data/datasets.py 또는 data/transforms.py 내부의 로직)
    """
    
    # [임시 Placeholder] 표준 ImageNet 전처리
    # TODO: 이 부분을 'texture', 'edge', 'sharpen'에 맞는
    #       실제 변환 로직으로 교체하세요.
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    
    # 예시: 'mode'에 따라 다른 변환을 적용해야 함
    if mode == 'edge':
        # print("경고: 'edge' 변환이 정의되지 않았습니다. 표준 변환을 사용합니다.")
        pass # 여기에 'edge' 변환 로직 추가
    elif mode == 'sharpen':
        # print("경고: 'sharpen' 변환이 정의되지 않았습니다. 표준 변환을 사용합니다.")
        pass # 여기에 'sharpen' 변환 로직 추가
    
    # 기본 (texture) 변환
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

# ==========================================================
# 헬퍼 함수 3: 앙상블 (제공된 코드)
# ==========================================================
def weighted_average_decision(results):
    """
    각 feature별 class * softmax 값을 평균내어 최종 확률로 사용.
    threshold=0.5 기준으로 class 결정.
    """
    total = 0.0
    n = len(results)
    if n == 0:
        return 0, 0.0
        
    for k, r in results.items():
        # 'class' (0 또는 1)와 'softmax' (확률)을 곱합니다.
        # (참고: 만약 r["class"]가 로짓(logit)이라면, 이 로직은 잘못되었습니다)
        # (r["class"]가 0 또는 1이라는 가정 하에 진행합니다)
        total += r["class"] * r["softmax"]
        
    # 이 로직은 "Fake일 확률"들의 가중 평균이 아니라,
    # "Fake(1)로 판정된 모델들의 확률 평균"에 가깝습니다.
    # 제공된 로직을 그대로 사용합니다.
    avg_score = total / n

    # 확률을 기반으로 최종 클래스 결정
    cls = 1 if avg_score >= 0.5 else 0
    return cls, avg_score

# ==========================================================
# 헬퍼 함수 4: 단일 이미지(프레임) 추론
# ==========================================================
@torch.no_grad()
def process_image(image_pil, models, transforms, device):
    """
    하나의 PIL 이미지를 3개 모델로 추론하고 결과를 앙상블합니다.
    """
    results = {}
    
    # 3개 모델(texture, edge, sharpen) 각각에 대해 추론 수행
    for mode in models.keys():
        model = models[mode]
        transform = transforms[mode]
        
        # 1. 이미지 변환
        img_tensor = transform(image_pil).unsqueeze(0).to(device)
        
        # 2. 모델 추론
        logit = model(img_tensor) # (1, 1)
        
        # 3. 결과 계산
        prob = torch.sigmoid(logit).item() # 확률 (0.0 ~ 1.0)
        cls = 1 if prob >= 0.5 else 0      # 클래스 (0 or 1)
        
        results[mode] = {"class": cls, "softmax": prob}

    # 4. 앙상블
    final_class, final_confidence = weighted_average_decision(results)
    
    return final_class, final_confidence, results

# ==========================================================
# 메인 실행 함수
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Model Ensemble Inference")
    parser.add_argument('--input', type=str, required=True, help="입력 이미지 또는 동영상 파일 경로")
    
    # --- [수정 필요] 3개 모델의 pth 파일 경로를 정확히 지정하세요 ---
    parser.add_argument('--model_texture', type=str, required=True, help="Texture 모델 .pth 파일 경로")
    parser.add_argument('--model_edge', type=str, required=True, help="Edge 모델 .pth 파일 경로")
    parser.add_argument('--model_sharpen', type=str, required=True, help="Sharpen 모델 .pth 파일 경로")
    # -----------------------------------------------------------
    
    parser.add_argument('--gpu_ids', type=str, default='0', help="GPU ID (e.g., 0 or 0,1,2)")
    args = parser.parse_args()

    # --- 1. 장치 설정 ---
    if torch.backends.mps.is_available() and args.gpu_ids:
        device = 'mps'
        print(f"⚡ Using Apple MPS (Mac GPU)", flush=True)
    elif torch.cuda.is_available() and args.gpu_ids:
        # This line is now fixed
        device = f'cuda:{args.gpu_ids.split(",")[0]}'
        print(f"⚡ Using CUDA: {device}", flush=True)
    else:
        device = 'cpu'
        print(f"⚡ Using CPU", flush=True)

    # --- 2. 모델 3개 로드 ---
    print("⌛ 모델 로드를 시작합니다...")
    models = {
        'texture': load_model(args.model_texture, device),
        'edge':    load_model(args.model_edge, device),
        'sharpen': load_model(args.model_sharpen, device)
    }

    # --- 3. 변환(Transform) 3개 준비 ---
    transforms = {
        'texture': get_transform('texture'),
        'edge':    get_transform('edge'),
        'sharpen': get_transform('sharpen')
    }

    # --- 4. 입력 파일 확장자 확인 ---
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
        return
        
    _, ext = os.path.splitext(input_path.lower())
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

    print("="*50)
    
    # --- 5. 이미지 추론 로직 ---
    if is_image:
        print(f"🖼️  이미지 추론 시작: {input_path}")
        try:
            image_pil = Image.open(input_path).convert("RGB")
            
            final_class, final_conf, details = process_image(image_pil, models, transforms, device)
            
            result_str = "FAKE" if final_class == 1 else "REAL"
            
            print("\n--- [개별 타워 결과] ---")
            for mode, res in details.items():
                print(f"  > {mode:<10}: Class={res['class']}, Prob={res['softmax']:.4f}")
            
            print("\n--- [최종 앙상블 결과] ---")
            print(f"  판정: {result_str} (Class: {final_class})")
            print(f"  신뢰도 (Confidence): {final_conf:.4f}")
            
        except Exception as e:
            print(f"❌ 이미지 처리 중 오류 발생: {e}")

    # --- 6. 동영상 추론 로직 ---
    elif is_video:
        print(f"🎬 동영상 추론 시작: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ 동영상을 열 수 없습니다.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_results = [] # (frame_class, frame_confidence) 저장

        pbar = tqdm(total=total_frames, desc="동영상 프레임 처리 중")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # OpenCV (BGR) -> PIL (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)
            
            # 각 프레임을 이미지처럼 앙상블 추론
            frame_class, frame_conf, _ = process_image(image_pil, models, transforms, device)
            
            frame_results.append((frame_class, frame_conf))
            pbar.update(1)
            
        cap.release()
        pbar.close()
        
        if not frame_results:
            print("❌ 동영상에서 프레임을 읽지 못했습니다.")
            return

        # --- 동영상 최종 판정 로직 ---
        video_final_class = 0 # REAL (0)로 기본값
        video_max_confidence = 0.0
        
        fake_frame_count = 0
        
        for (cls, conf) in frame_results:
            # 한 프레임이라도 Fake(1)이면 동영상은 Fake(1)
            if cls == 1:
                video_final_class = 1
                fake_frame_count += 1
            
            # 동영상의 최종 신뢰도는 모든 프레임 중 가장 높은 신뢰도 값
            if conf > video_max_confidence:
                video_max_confidence = conf
                
        result_str = "FAKE" if video_final_class == 1 else "REAL"

        print("\n--- [최종 동영상 판정] ---")
        print(f"  총 프레임: {len(frame_results)}")
        print(f"  Fake 판정 프레임: {fake_frame_count} 개")
        print(f"  판정: {result_str} (Class: {video_final_class})")
        print(f"  최대 신뢰도 (Max Confidence): {video_max_confidence:.4f}")

    else:
        print(f"❌ 지원하지 않는 파일 형식입니다: {ext}")

    print("="*50)


if __name__ == "__main__":
    main()