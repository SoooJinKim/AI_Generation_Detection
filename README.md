# AI Generation Detection

AI 생성 콘텐츠(이미지/동영상) 탐지를 위한 딥러닝 프레임워크

## 📋 Features

- ✅ **Single-Tower 학습**: texture, edge, sharpen transformation 지원
- ✅ **이미지 + 동영상 처리**: 자동으로 파일 타입 감지 및 처리
- ✅ **CPU/GPU 지원**: 자동으로 사용 가능한 디바이스 감지
- 🚧 **Multi-Tower**: 여러 transformation 결합 (TODO)

## 🚀 Quick Start

### 1. 환경 설정 (uv 사용)

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 클론
git clone <repository-url>
cd AI_Generation_Detection

# Python 환경 생성 및 패키지 설치
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 2. 데이터셋 준비

```
dataset/
├── train/
│   ├── real/       # 진짜 이미지/동영상 (label=0)
│   │   ├── img1.jpg
│   │   ├── video1.mp4
│   │   └── ...
│   └── fake/       # 가짜 이미지/동영상 (label=1)
│       ├── img1.jpg
│       ├── video1.mp4
│       └── ...
└── val/
    ├── real/
    └── fake/
```

**지원 파일 형식:**
- 이미지: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
- 동영상: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`

**Label 정의:**
- `0` = Real (진짜)
- `1` = Fake (가짜)

### 3. 학습

```bash
python train.py \
    --tower_mode single \
    --mode realfake \
    --dataroot ./dataset \
    --train_split train \
    --val_split val \
    --transform_mode texture \
    --batch_size 32 \
    --lr 0.0001 \
    --niter 50 \
    --name my_experiment
```

**빠른 실행:**
```bash
# CPU 환경
bash example_single_tower_cpu.sh

# GPU 환경
bash example_single_tower.sh
```

## 📚 Arguments

### 필수 Arguments

| Argument | 설명 | 기본값 | 예시 |
|----------|------|--------|------|
| `--tower_mode` | Tower 모드 | `single` | `single`, `multi` (TODO) |
| `--mode` | 데이터셋 모드 | `binary` | `realfake` |
| `--dataroot` | 데이터셋 루트 경로 | `./dataset/` | `./dataset` |
| `--train_split` | 학습 데이터 폴더 | `train` | `train` |
| `--val_split` | 검증 데이터 폴더 | `val` | `val` |

### Transform Arguments

| Argument | 설명 | 기본값 | 선택 |
|----------|------|--------|------|
| `--transform_mode` | Single-Tower transform | `texture` | `texture`, `edge`, `sharpen` |
| `--transform_modes` | Multi-Tower transforms | `None` | `texture,edge,sharpen` (TODO) |
| `--sample_video_frame` | 동영상 프레임 샘플링 | `random` | `random`, `first`, `middle` |

**Transform 종류:**
- `texture`: Sobel magnitude (텍스처 강조) - 범용적
- `edge`: Canny edge detection - GAN 탐지에 효과적
- `sharpen`: Sharpening filter - Deepfake 탐지에 효과적

### 학습 Arguments

| Argument | 설명 | 기본값 |
|----------|------|--------|
| `--batch_size` | 배치 크기 | `64` |
| `--lr` | Learning rate | `0.0001` |
| `--optim` | Optimizer | `adam` |
| `--niter` | 학습 에폭 수 | `1000` |
| `--delr_freq` | LR 감소 주기 | `20` |
| `--earlystop_epoch` | Early stopping patience | `15` |
| `--name` | 실험 이름 | `experiment_name` |

### GPU Arguments

| Argument | 설명 | 기본값 |
|----------|------|--------|
| `--gpu_ids` | GPU ID (쉼표로 구분) | `0` |
| `--num_threads` | DataLoader worker 수 | `8` |

**GPU 사용:**
- GPU 있음: `--gpu_ids 0` 또는 `--gpu_ids 0,1` (multi-GPU)
- CPU만: `--gpu_ids -1`

### Video Arguments

| Argument | 설명 | 기본값 | 선택 |
|----------|------|--------|------|
| `--video_num_frames` | 동영상당 샘플링 프레임 수 | `5` | `1~N` |
| `--video_sampling` | 프레임 샘플링 방식 | `uniform` | `uniform`, `random` |
| `--video_voting` | 예측 voting 방식 | `max` | `max`, `avg`, `majority` |

**Voting 방식:**
- `max`: 최대 확률값 사용 (하나라도 fake면 fake)
- `avg`: 평균 확률값 사용
- `majority`: 과반수 투표 (0.5 threshold)

## 📖 사용 예제

### 1. Texture transform으로 학습

```bash
python train.py \
    --tower_mode single \
    --mode realfake \
    --dataroot ./dataset \
    --train_split train \
    --val_split val \
    --transform_mode texture \
    --batch_size 32 \
    --niter 100 \
    --name texture_experiment
```

### 2. Edge transform으로 학습

```bash
python train.py \
    --tower_mode single \
    --transform_mode edge \
    --dataroot ./dataset \
    --batch_size 32 \
    --name edge_experiment
```

### 3. Multi-GPU 학습

```bash
python train.py \
    --tower_mode single \
    --transform_mode texture \
    --dataroot ./dataset \
    --batch_size 64 \
    --gpu_ids 0,1 \
    --name multi_gpu_experiment
```

### 4. CPU 학습 (GPU 없는 환경)

```bash
python train.py \
    --tower_mode single \
    --transform_mode texture \
    --dataroot ./dataset \
    --batch_size 8 \
    --gpu_ids -1 \
    --name cpu_experiment
```

## 🎥 동영상 처리

### 학습 시
- 동영상에서 **1개 프레임만 샘플링** (빠른 학습)
- `--sample_video_frame` 옵션으로 샘플링 방법 선택

### 검증 시
- 동영상에서 **여러 프레임 샘플링** (기본 5개)
- 각 프레임 개별 예측 → 하나라도 fake면 fake로 판단

```bash
# 동영상 검증 스크립트
python validate_video.py \
    --model_path ./checkpoints/model.pth \
    --dataroot ./dataset/val \
    --transform_mode texture
```

## 🔧 주요 파일 구조

```
AI_Generation_Detection/
├── train.py                    # 학습 스크립트
├── validate.py                 # 이미지 검증
├── validate_video.py           # 동영상 검증
├── networks/
│   ├── trainer.py              # 모듈화된 Trainer
│   ├── resnet.py               # NPR 커스텀 ResNet
│   └── multi_tower.py          # Multi-Tower 모델 (TODO)
├── data/
│   ├── datasets.py             # 데이터셋 클래스
│   └── __init__.py             # 데이터로더
├── options/
│   ├── base_options.py         # 기본 옵션
│   ├── train_options.py        # 학습 옵션
│   └── test_options.py         # 테스트 옵션
└── example_single_tower*.sh    # 실행 예제
```

## ⚠️ 주의사항

1. **Label 정의**: `0=Real`, `1=Fake`
2. **동영상 학습**: 학습 시에는 프레임 1개만 사용 (효율성)
3. **동영상 검증**: 검증 시에는 여러 프레임 사용 (정확도)
5. **GPU 메모리**: 배치 크기를 GPU 메모리에 맞게 조정

## 📝 TODO

- [ ] Multi-Tower 아키텍처 구현
- [ ] Feature fusion 방법 (concat/average/attention) 통합
- [ ] 추가 transformation 지원
- [ ] Pre-trained model 제공

## 📄 License

MIT License

## 🙏 Acknowledgments

This project is based on the NPR (CVPR 2024) paper architecture.
