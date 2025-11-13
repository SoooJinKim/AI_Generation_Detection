import os
import sys
import time
import torch
import torch.nn as nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from networks.resnet import resnet50
# from networks.multi_tower import MultiTowerModel, SingleTowerWrapper # <- 여기서 MultiTowerModel을 직접 정의
from networks.multi_tower import SingleTowerWrapper
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
from tqdm import tqdm
from torchvision import transforms as T
import random
import copy # 멀티 타워 모델 복제(clone)에 필요

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

def save_side_by_side(tensor_img, orig_path, mode="sharpen"):
    global SAVE_COUNT
    if SAVE_COUNT >= MAX_SAVE:
        return
    try:
        # 동영상 파일인지 확인
        ext = os.path.splitext(orig_path)[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']:
            import cv2
            cap = cv2.VideoCapture(orig_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                orig = Image.fromarray(frame)
            else:
                print(f"[WARN] Cannot read video frame: {orig_path}", flush=True)
                return
        else:
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
        print(f"✅ 샘플 저장: {save_path}", flush=True)
        SAVE_COUNT += 1
    except Exception as e:
        print(f"[WARN] sample save failed for {orig_path}: {e}", flush=True)

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
    print(f"💾 Saved state_dict: {path}", flush=True)

# ==========================================================
# [신규] 멀티 타워 모델 및 융합(Fusion) 헤드 정의
# (요청하신 Aggregator의 "학습 가능한" 버전입니다)
# ==========================================================
class MultiTowerModel(nn.Module):
    def __init__(self, transform_modes, fusion_method='concat', num_classes=1, embedding_dim=2048):
        """
        학습 가능한 멀티 타워 모델.
        
        Args:
            transform_modes (list): ['texture', 'edge', 'sharpen']
            fusion_method (str): 'concat', 'max', 'weighted'
        """
        super().__init__()
        self.transform_modes = transform_modes
        self.fusion_method = fusion_method
        self.num_towers = len(transform_modes)
        
        print(f"[MultiTowerModel] 융합(Fusion) 방식: {self.fusion_method}")

        # 1. 각 피처(transform_mode)에 대한 타워 생성
        base_model_instance = resnet50(pretrained=False, num_classes=num_classes)
        self.towers = nn.ModuleDict()
        for mode in self.transform_modes:
            # SingleTowerWrapper가 get_embedding()을 제공한다고 가정
            self.towers[mode] = SingleTowerWrapper(copy.deepcopy(base_model_instance))
            
        # 2. Aggregator (Fusion Head) 생성
        # ResNet50의 기본 embedding_dim은 2048입니다.
        if self.fusion_method == 'concat':
            # 모든 타워의 임베딩을 연결
            fusion_input_dim = embedding_dim * self.num_towers
            self.fusion_head = nn.Sequential(
                nn.Linear(fusion_input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
            print(f"[MultiTowerModel] Concat 퓨전 헤드 생성 (Input: {fusion_input_dim} -> 1)")
        
        elif self.fusion_method in ['weighted', 'average']:
            # 각 타워의 로짓(logit)을 가중 평균
            # (B, num_towers) -> (B, 1)
            self.fusion_head = nn.Linear(self.num_towers, num_classes)
            print(f"[MultiTowerModel] Weighted 퓨전 헤드 생성 (Input: {self.num_towers} -> 1)")
            
        elif self.fusion_method == 'max':
            # 'max'는 학습 가능한 파라미터가 없는 연산
            self.fusion_head = None
            print(f"[MultiTowerModel] Max 퓨전 사용 (파라미터 없음)")
        
        else:
            raise ValueError(f"지원되지 않는 퓨전 방식입니다: {self.fusion_method}")
            
    def forward(self, data_dict):
        """
        Multi-tower 모델은 데이터로더로부터 딕셔너리 입력을 받습니다.
        Input:
            data_dict (dict): {'texture': (B,C,H,W), 'edge': (B,C,H,W), ...}
        Output:
            final_logits (Tensor): (B,) Trainer가 사용할 최종 로짓
        """
        
        # 'concat' 모드는 임베딩을 사용하고, 나머지는 로짓을 사용
        if self.fusion_method == 'concat':
            embeddings = []
            for mode in self.transform_modes:
                x = data_dict[mode]
                # get_embedding() 메서드가 SingleTowerWrapper에 구현되어 있어야 함
                try:
                    # (B, E)
                    emb = self.towers[mode].get_embedding(x) 
                    embeddings.append(emb)
                except AttributeError:
                    print("오류: SingleTowerWrapper에 get_embedding() 메서드가 없습니다.", flush=True)
                    print("대안: 'concat' 대신 'weighted' 퓨전 방식을 사용하세요.", flush=True)
                    raise
            
            # (B, E * num_towers)
            fused_embedding = torch.cat(embeddings, dim=1)
            # (B, 1)
            final_logits = self.fusion_head(fused_embedding)
            
        else:
            # 'weighted' 또는 'max' 모드 (로짓 기반 융합)
            tower_logits = []
            for mode in self.transform_modes:
                x = data_dict[mode]
                # (B, 1)
                logit = self.towers[mode](x)
                tower_logits.append(logit)
            
            # (B, num_towers)
            all_logits = torch.cat(tower_logits, dim=1)
            
            if self.fusion_method in ['weighted', 'average']:
                # (B, 1)
                final_logits = self.fusion_head(all_logits)
            elif self.fusion_method == 'max':
                # (B, 1)
                final_logits, _ = torch.max(all_logits, dim=1, keepdim=True)
        
        # (B, 1) -> (B,)
        return final_logits.squeeze(1)


# ------------------------
# 검증 옵션
# ------------------------
vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    
    if val_opt.val_split and val_opt.val_split.strip():
        val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    else:
        val_opt.dataroot = val_opt.dataroot if val_opt.dataroot.endswith('/') else val_opt.dataroot + '/'
    
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    # --- [수정된 부분] ---
    # macOS에서 발생하는 lambda pickle 오류를 막기 위해
    # 검증(validation)은 메인 스레드에서 실행합니다. (0으로 설정)
    val_opt.num_threads = 0 
    # --- [수정 완료] ---
    
    return val_opt

# ------------------------
# [수정] 모델 생성 함수
# ------------------------
def create_model(opt):
    """
    옵션에 따라 Single-Tower 또는 Multi-Tower 모델 생성
    """
    tower_mode = getattr(opt, 'tower_mode', 'single')
    
    if tower_mode == 'multi':
        # Multi-Tower 모델 (새로 구현)
        print("=" * 60, flush=True)
        print("🏗️  Multi-Tower Model (학습 가능 버전)", flush=True)
        print("=" * 60, flush=True)
        
        if not hasattr(opt, 'transform_modes') or not opt.transform_modes:
            raise ValueError("--tower_mode 'multi'를 사용하려면 --transform_modes가 필요합니다.")
            
        transform_modes = opt.transform_modes.split(',')
        fusion_method = getattr(opt, 'fusion_method', 'concat') # 예: --fusion_method concat
        
        print(f"  Transform modes: {transform_modes}", flush=True)
        print(f"  Fusion method: {fusion_method}", flush=True)
        print("=" * 60 + "\n", flush=True)

        model = MultiTowerModel(
            transform_modes=transform_modes,
            fusion_method=fusion_method,
            num_classes=1
        )
        return model
    
    elif tower_mode == 'single':
        # Single-Tower 모델 (기존 로직)
        transform_mode = getattr(opt, 'transform_mode', 'sharpen')
        
        print("\n" + "=" * 60, flush=True)
        print("🏗️  Single-Tower Model", flush=True)
        print("=" * 60, flush=True)
        print(f"  Architecture: ResNet50", flush=True)
        print(f"  Transform mode: {transform_mode}", flush=True)
        print(f"  Training from scratch (no pretrained)", flush=True)
        print(f"  Dataset: {opt.dataroot}", flush=True)
        print("=" * 60 + "\n", flush=True)
        
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
    
    if opt.train_split and opt.train_split.strip():
        opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    else:
        opt.dataroot = opt.dataroot if opt.dataroot.endswith('/') else opt.dataroot + '/'
    
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)), flush=True)
    
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    
    # [중요] 멀티 타워 모드일 때, 데이터로더가 dict를 반환하도록 설정
    # (data/__init__.py의 create_dataloader가 이 옵션을 확인해야 함)
    if getattr(opt, 'tower_mode', 'single') == 'multi':
        print("[INFO] 멀티 타워 모드: 데이터로더가 딕셔너리 형태('dict')로 데이터를 반환해야 합니다.", flush=True)
        # 이 옵션이 data.py로 전달되어야 함
        opt.dataset_mode = 'dict' 
        
    data_loader = create_dataloader(opt)
    
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    model = create_model(opt)
    
    if len(opt.gpu_ids) > 1 and torch.cuda.is_available():
        print(f"⚡ Using {len(opt.gpu_ids)} GPUs with DataParallel", flush=True)
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    
    # [수정] Mac(mps) 장치 감지 로직 추가
    if torch.backends.mps.is_available() and opt.gpu_ids:
        device = 'mps'
        print("⚡ Using Apple MPS (Mac GPU)", flush=True)
    elif len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        device = f'cuda:{opt.gpu_ids[0]}'
        print(f"⚡ Using CUDA: {device}", flush=True)
    else:
        device = 'cpu'
        print("⚡ Using CPU", flush=True)
    
    trainer = Trainer(
        model=model,
        device=device,
        lr=opt.lr,
        optim=opt.optim,
        beta1=opt.beta1
    )
    
    trainer.train()
    print(f'cwd: {os.getcwd()}', flush=True)

    ckpt_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    patience_limit = int(getattr(opt, "patience", getattr(opt, "earlystop_epoch", 0)))
    best_metric = -float("inf")
    best_epoch  = 0
    stale_count = 0
    stopped_early = False
    VAL_EVERY = 5
    MONITOR = "AP"
    
    # [수정] 멀티 타워 플래그
    IS_MULTI_TOWER = getattr(opt, 'tower_mode', 'single') == 'multi'

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        epoch_iter = 0
        loss_value = 0.0 # loss_value 초기화

        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}/{opt.niter}")
        for i, batch in pbar:
            
            log_str = "" # 로그 문자열 초기화
            
            # [수정] 멀티 타워 학습 루프
            if IS_MULTI_TOWER:
                # data_dict = {'texture': (B,C,H,W), 'edge': (B,C,H,W), ...}
                if len(batch) == 4:
                    data_dict, labels, paths, is_video_flags = batch
                elif len(batch) == 3:
                    data_dict, labels, paths = batch
                    is_video_flags = None
                else:
                    data_dict, labels = batch
                    paths, is_video_flags = None, None
                
                if is_video_flags and any(is_video_flags):
                    print("[WARN] Multi-tower video training not supported. Skipping batch.", flush=True)
                    continue
                    
                trainer.total_steps += 1
                epoch_iter += opt.batch_size
                
                trainer.set_input((data_dict, labels))
                loss_value = trainer.optimize_parameters()
                log_str = f"loss: {loss_value:.4f} | step: {trainer.total_steps} | lr: {trainer.lr:.6f} [Multi]"

            else:
                # ---- 기존 Single-Tower 학습 루프 ----
                if len(batch) == 4:
                    data, labels, paths, is_video_flags = batch
                elif len(batch) == 3:
                    data, labels, paths, is_video_flags = None
                else:
                    data, labels = batch
                    paths, is_video_flags = None, None

                if epoch == 0 and paths is not None and SAVE_COUNT < MAX_SAVE:
                    if not isinstance(data, list) and not isinstance(data[0], list):
                        for b in range(min(data.size(0), MAX_SAVE - SAVE_COUNT)):
                            save_side_by_side(data[b], paths[b], mode=getattr(opt, "transform_mode", "edge"))

                if is_video_flags and any(is_video_flags):
                    for sample_idx, is_video in enumerate(is_video_flags):
                        if is_video:
                            frames, label = data[sample_idx], labels[sample_idx]
                            for frame in frames:
                                trainer.total_steps += 1
                                trainer.set_input((frame.unsqueeze(0), label.unsqueeze(0) if label.dim() == 0 else label))
                                loss_value = trainer.optimize_parameters()
                                log_str = f"loss: {loss_value:.4f} | step: {trainer.total_steps} | lr: {trainer.lr:.6f} | video"
                                if trainer.total_steps % opt.loss_freq == 0:
                                    train_writer.add_scalar('loss', loss_value, trainer.total_steps)
                        else:
                            trainer.total_steps += 1
                            trainer.set_input((data[sample_idx].unsqueeze(0), labels[sample_idx].unsqueeze(0) if labels[sample_idx].dim() == 0 else labels[sample_idx]))
                            loss_value = trainer.optimize_parameters()
                            log_str = f"loss: {loss_value:.4f} | step: {trainer.total_steps} | lr: {trainer.lr:.6f}"
                            if trainer.total_steps % opt.loss_freq == 0:
                                train_writer.add_scalar('loss', loss_value, trainer.total_steps)
                    pbar.set_postfix_str(log_str) # 비디오 처리 후 tqdm 업데이트
                else:
                    trainer.total_steps += 1
                    epoch_iter += opt.batch_size
                    trainer.set_input((data, labels))
                    loss_value = trainer.optimize_parameters()
                    log_str = f"loss: {loss_value:.4f} | step: {trainer.total_steps} | lr: {trainer.lr:.6f}"
            
            # [수정] tqdm 진행률 표시줄을 매 스텝 갱신
            if log_str:
                pbar.set_postfix_str(log_str)
            # TensorBoard 로깅 (기존 opt.loss_freq 유지)
            if trainer.total_steps % opt.loss_freq == 0 and loss_value != 0.0:
                 train_writer.add_scalar('loss', loss_value, trainer.total_steps)


        # Learning rate 조정
        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                  f'changing lr at the end of epoch {epoch}, iters {trainer.total_steps}', flush=True)
            trainer.adjust_learning_rate()
            
        # ---- 5에폭마다 검증 ----
        if (epoch+1) % VAL_EVERY == 0:
            trainer.eval()
            eval_model = trainer.model.module if isinstance(trainer.model, torch.nn.DataParallel) else trainer.model
            
            # [중요] 멀티 타워 모드일 때, validate.py도 dict 입력을 처리해야 함
            acc, ap = validate(eval_model, val_opt)[:2]
            val_writer.add_scalar('accuracy', acc, trainer.total_steps)
            val_writer.add_scalar('ap', ap, trainer.total_steps)
            print(f"(Val @ epoch {epoch+1}) acc: {acc:.6f}; ap: {ap:.6f}", flush=True)

            save_state_dict_only(trainer.model, ckpt_dir, f"model_epoch_{epoch+1}.pth")

            metric = ap
            if metric > best_metric + 1e-8:
                best_metric = metric
                best_epoch = epoch + 1
                stale_count = 0
                save_state_dict_only(trainer.model, ckpt_dir, "model_epoch_best.pth")
                print(f"🔥 New best {MONITOR}: {best_metric:.6f} @ epoch {best_epoch}", flush=True)
            else:
                stale_count += 1
                print(f"⏳ No {MONITOR} improvement ({stale_count}/{patience_limit})", flush=True)
                if patience_limit > 0 and stale_count >= patience_limit:
                    print(f"🛑 Early stopping triggered "
                          f"(no {MONITOR} improvement for {patience_limit} validations).", flush=True)
                    save_state_dict_only(trainer.model, ckpt_dir, f"model_epoch_{epoch+1}_earlystop.pth")
                    stopped_early = True
                    break
            trainer.train()

        if stopped_early:
            break

    if not stopped_early:
        save_state_dict_only(trainer.model, ckpt_dir, "model_epoch_last.pth")

    print(f"✅ Training finished. Best {MONITOR}: {best_metric:.6f} @ epoch {best_epoch}", flush=True)