import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from networks.resnet import resnet50
from networks.multi_tower import MultiTowerModel, SingleTowerWrapper
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
from tqdm import tqdm
from torchvision import transforms as T
import random

# ------------------------
# 유틸: 샘플 저장
# ------------------------
SAVE_COUNT = 0
MAX_SAVE = 20
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def denorm_imgnet(t):
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device).view(-1,1,1)
    std  = torch.tensor(IMAGENET_STD,  dtype=t.dtype, device=t.device).view(-1,1,1)
    x = t*std + mean
    return x.clamp(0,1)

to_pil = T.ToPILImage()

def save_side_by_side(tensor_img, orig_path, mode="texture"):
    global SAVE_COUNT
    if SAVE_COUNT >= MAX_SAVE:
        return
    try:
        # 동영상 파일인지 확인
        ext = os.path.splitext(orig_path)[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']:
            # 동영상은 첫 프레임을 추출해서 저장
            import cv2
            cap = cv2.VideoCapture(orig_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                orig = Image.fromarray(frame)
            else:
                print(f"[WARN] Cannot read video frame: {orig_path}")
                return
        else:
            # 이미지 파일
            orig = Image.open(orig_path).convert("RGB")
        
        W, H = orig.size
        den = denorm_imgnet(tensor_img.detach().cpu())
        transformed = to_pil(den).resize((W, H))

        combined = Image.new("RGB", (W*2, H))
        combined.paste(orig, (0, 0))
        combined.paste(transformed, (W, 0))

        base = os.path.splitext(os.path.basename(orig_path))[0]
        out_dir = os.path.join("samples", base)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{mode}_comparison.png")
        combined.save(save_path)
        print(f"✅ 샘플 저장: {save_path}")
        SAVE_COUNT += 1
    except Exception as e:
        print(f"[WARN] sample save failed for {orig_path}: {e}")

# ------------------------
# 시드 고정
# ------------------------
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

# ------------------------
# 체크포인트 저장 (state_dict만)
# ------------------------
def save_state_dict_only(model, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    sd = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    path = os.path.join(save_dir, filename)
    torch.save(sd, path)
    print(f"💾 Saved state_dict: {path}")

# ------------------------
# 검증 옵션
# ------------------------
vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    
    # val_split이 지정되어 있으면 경로에 추가
    if val_opt.val_split and val_opt.val_split.strip():
        val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    else:
        # val_split이 없으면 dataroot를 그대로 사용
        val_opt.dataroot = val_opt.dataroot if val_opt.dataroot.endswith('/') else val_opt.dataroot + '/'
    
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    return val_opt

# ------------------------
# 모델 생성 함수
# ------------------------
def create_model(opt):
    """
    옵션에 따라 Single-Tower 또는 Multi-Tower 모델 생성
    
    Args:
        opt: TrainOptions
        
    Returns:
        model: nn.Module
    """
    tower_mode = getattr(opt, 'tower_mode', 'single')
    
    if tower_mode == 'multi':
        # Multi-Tower 모델 (NOT IMPLEMENTED)
        print("=" * 60)
        print("🚧 Multi-Tower Architecture - NOT IMPLEMENTED")
        print("=" * 60)
        
        if hasattr(opt, 'transform_modes') and opt.transform_modes:
            print(f"Requested transform modes: {opt.transform_modes}")
        else:
            print(f"Requested transform modes: Not specified")
        
        print(f"Fusion method: {getattr(opt, 'fusion_method', 'concat')}")
        print("\n⚠️  Multi-Tower is NOT IMPLEMENTED yet!")
        print("\nTODO List:")
        print("  1. Multiple transformation pipelines")
        print("  2. Feature fusion (concat/average/attention)")
        print("  3. Unified output generation")
        print("  4. Multi-tower specific training logic")
        print("\n💡 Use --tower_mode single for now")
        print("=" * 60)
        raise NotImplementedError("Multi-Tower training is not implemented yet. Use --tower_mode single")
    
    elif tower_mode == 'single':
        # Single-Tower 모델
        transform_mode = getattr(opt, 'transform_mode', 'texture')
        
        print("\n" + "=" * 60)
        print("🏗️  Single-Tower Model")
        print("=" * 60)
        print(f"  Architecture: ResNet50")
        print(f"  Transform mode: {transform_mode}")
        print(f"  Training from scratch (no pretrained)")
        print(f"  Dataset: {opt.dataroot}")
        print("=" * 60 + "\n")
        
        model = resnet50(pretrained=False, num_classes=1)
        model = SingleTowerWrapper(model)
        return model
    
    else:
        raise ValueError(f"Unknown tower_mode: {tower_mode}. Use 'single' or 'multi'")


# ------------------------
# main
# ------------------------
if __name__ == '__main__':
    opt = TrainOptions().parse()
    seed_torch(100)
    
    # 데이터 경로 설정
    Testdataroot = os.path.join(opt.dataroot, 'test')
    
    # train_split이 지정되어 있으면 경로에 추가
    if opt.train_split and opt.train_split.strip():
        opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    else:
        # train_split이 없으면 dataroot를 그대로 사용
        opt.dataroot = opt.dataroot if opt.dataroot.endswith('/') else opt.dataroot + '/'
    
    # 로거 초기화
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)))
    
    # Validation 옵션
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    
    # 데이터로더 생성
    data_loader = create_dataloader(opt)
    
    # TensorBoard writer
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    # 모델 생성
    model = create_model(opt)
    
    # DataParallel 적용
    if len(opt.gpu_ids) > 1 and torch.cuda.is_available():
        print(f"⚡ Using {len(opt.gpu_ids)} GPUs with DataParallel")
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    
    # Trainer 생성
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        device = f'cuda:{opt.gpu_ids[0]}'
    else:
        device = 'cpu'
    
    trainer = Trainer(
        model=model,
        device=device,
        lr=opt.lr,
        optim=opt.optim,
        beta1=opt.beta1
    )
    
    trainer.train()
    print(f'cwd: {os.getcwd()}')

    # 체크포인트 디렉토리
    ckpt_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Early Stopping 설정 ----
    patience_limit = int(getattr(opt, "patience", getattr(opt, "earlystop_epoch", 0)))
    best_metric = -float("inf")
    best_epoch  = 0
    stale_count = 0
    stopped_early = False
    VAL_EVERY = 5
    MONITOR = "AP"

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        epoch_iter = 0

        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}/{opt.niter}")
        for i, batch in pbar:
            # 배치 언패킹 (동영상 지원)
            if len(batch) == 4:
                # (images_or_frames, labels, paths, is_video_flags)
                data, labels, paths, is_video_flags = batch
            elif len(batch) == 3:
                data, labels, paths = batch
                is_video_flags = None
            else:
                data, labels = batch
                paths = None
                is_video_flags = None

            # 첫 에폭에서 샘플 저장
            if epoch == 0 and paths is not None and SAVE_COUNT < MAX_SAVE:
                if not isinstance(data, list) and not isinstance(data[0], list):
                    # 일반 이미지 배치
                    for b in range(min(data.size(0), MAX_SAVE - SAVE_COUNT)):
                        save_side_by_side(data[b], paths[b], mode=getattr(opt, "transform_mode", "texture"))

            # 동영상 처리: 각 프레임을 하나씩 학습
            if is_video_flags and any(is_video_flags):
                for sample_idx, is_video in enumerate(is_video_flags):
                    if is_video:
                        # 동영상: 프레임 리스트
                        frames = data[sample_idx]  # List[Tensor]
                        label = labels[sample_idx]
                        
                        # 각 프레임으로 학습
                        for frame in frames:
                            trainer.total_steps += 1
                            frame_batch = frame.unsqueeze(0)  # [1, C, H, W]
                            label_batch = label.unsqueeze(0) if label.dim() == 0 else label
                            
                            trainer.set_input((frame_batch, label_batch))
                            loss_value = trainer.optimize_parameters()
                            
                            if trainer.total_steps % opt.loss_freq == 0:
                                log_str = f"loss: {loss_value:.4f} | step: {trainer.total_steps} | lr: {trainer.lr:.6f} | video"
                                pbar.set_postfix_str(log_str)
                                train_writer.add_scalar('loss', loss_value, trainer.total_steps)
                    else:
                        # 이미지: 일반 처리
                        trainer.total_steps += 1
                        img_batch = data[sample_idx].unsqueeze(0)
                        label_batch = labels[sample_idx].unsqueeze(0) if labels[sample_idx].dim() == 0 else labels[sample_idx]
                        
                        trainer.set_input((img_batch, label_batch))
                        loss_value = trainer.optimize_parameters()
                        
                        if trainer.total_steps % opt.loss_freq == 0:
                            log_str = f"loss: {loss_value:.4f} | step: {trainer.total_steps} | lr: {trainer.lr:.6f}"
                            pbar.set_postfix_str(log_str)
                            train_writer.add_scalar('loss', loss_value, trainer.total_steps)
            else:
                # 일반 이미지 배치 처리
                trainer.total_steps += 1
                epoch_iter += opt.batch_size
                
                trainer.set_input((data, labels))
                loss_value = trainer.optimize_parameters()

                if trainer.total_steps % opt.loss_freq == 0:
                    log_str = f"loss: {loss_value:.4f} | step: {trainer.total_steps} | lr: {trainer.lr:.6f}"
                    pbar.set_postfix_str(log_str)
                    train_writer.add_scalar('loss', loss_value, trainer.total_steps)

        # Learning rate 조정
        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                  f'changing lr at the end of epoch {epoch}, iters {trainer.total_steps}')
            trainer.adjust_learning_rate()
            
        # ---- 5에폭마다 검증 ----
        if (epoch+1) % VAL_EVERY == 0:
            trainer.eval()
            # validate 함수에 실제 모델 전달
            eval_model = trainer.model.module if isinstance(trainer.model, torch.nn.DataParallel) else trainer.model
            acc, ap = validate(eval_model, val_opt)[:2]
            val_writer.add_scalar('accuracy', acc, trainer.total_steps)
            val_writer.add_scalar('ap', ap, trainer.total_steps)
            print(f"(Val @ epoch {epoch+1}) acc: {acc:.6f}; ap: {ap:.6f}")

            # 현재 에폭 저장
            save_state_dict_only(trainer.model, ckpt_dir, f"model_epoch_{epoch+1}.pth")

            # 모니터 지표: ap
            metric = ap
            if metric > best_metric + 1e-8:
                best_metric = metric
                best_epoch = epoch + 1
                stale_count = 0
                save_state_dict_only(trainer.model, ckpt_dir, "model_epoch_best.pth")
                print(f"🔥 New best {MONITOR}: {best_metric:.6f} @ epoch {best_epoch}")
            else:
                stale_count += 1
                print(f"⏳ No {MONITOR} improvement ({stale_count}/{patience_limit})")
                if patience_limit > 0 and stale_count >= patience_limit:
                    print(f"🛑 Early stopping triggered "
                          f"(no {MONITOR} improvement for {patience_limit} validations).")
                    save_state_dict_only(trainer.model, ckpt_dir, f"model_epoch_{epoch+1}_earlystop.pth")
                    stopped_early = True
                    break
            trainer.train()

        if stopped_early:
            break

    # 마지막 저장
    if not stopped_early:
        save_state_dict_only(trainer.model, ckpt_dir, "model_epoch_last.pth")

    print(f"✅ Training finished. Best {MONITOR}: {best_metric:.6f} @ epoch {best_epoch}")