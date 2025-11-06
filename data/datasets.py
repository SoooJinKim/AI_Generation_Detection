import os
import cv2
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.data import Dataset
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Supported video extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

SAVE_COUNT = 0
MAX_SAVE = 20

def save_with_original(tensor, pil_img, path, mode="texture"):
    """
    tensor: transformed tensor [C,H,W]
    pil_img: 원본 PIL.Image
    path: 원래 파일 경로
    mode: 변환 모드 (texture, edge, sharpen 등)
    """
    global SAVE_COUNT
    if SAVE_COUNT >= MAX_SAVE:
        return tensor

    # Tensor → PIL 변환
    transformed = transforms.ToPILImage()(tensor.cpu().clone().detach())

    # 크기 맞추기 (원본/변환 이미지 크기 동일화)
    W, H = pil_img.size
    transformed = transformed.resize((W, H))

    # 두 이미지 붙이기 (좌우)
    combined = Image.new("RGB", (W * 2, H))
    combined.paste(pil_img, (0, 0))
    combined.paste(transformed, (W, 0))

    # 저장 디렉토리
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join("samples", base)
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, f"{mode}_comparison.png")
    combined.save(save_path)

    print(f"✅ 샘플 저장: {save_path}")
    SAVE_COUNT += 1
    return tensor

def load_video_frame(video_path, frame_idx=0):
    """동영상에서 특정 프레임을 추출"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")
    
    # frame_idx가 -1이면 랜덤 프레임
    if frame_idx == -1:
        frame_idx = np.random.randint(0, total_frames)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Cannot read frame {frame_idx} from video: {video_path}")
    
    # BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def load_video_frames(video_path, num_frames=5, sampling='uniform'):
    """
    동영상에서 여러 프레임을 샘플링
    
    Args:
        video_path: 동영상 경로
        num_frames: 샘플링할 프레임 수
        sampling: 'uniform' (균등 간격) 또는 'random' (랜덤)
    
    Returns:
        List of PIL Images
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")
    
    # 실제 샘플링할 프레임 수 (전체 프레임 수보다 많으면 조정)
    num_frames = min(num_frames, total_frames)
    
    # 프레임 인덱스 결정
    if sampling == 'uniform':
        # 균등 간격으로 샘플링
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:  # random
        # 랜덤 샘플링
        frame_indices = np.random.choice(total_frames, num_frames, replace=False)
        frame_indices = sorted(frame_indices)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from video: {video_path}")
    
    return frames


class RealFakeDataset(Dataset):
    """
    dataset/{real, fake} 구조를 지원하는 데이터셋
    이미지와 동영상 모두 처리 가능
    Multi-Tower를 고려하여 multiple transforms를 적용할 수 있음
    """
    def __init__(self, root, transforms_list=None, is_train=True, video_num_frames=5, video_sampling='uniform'):
        """
        Args:
            root: dataset 루트 디렉토리 (real, fake 폴더가 있는)
            transforms_list: List of transforms or single transform
            is_train: 학습 모드 여부
            video_num_frames: 동영상에서 샘플링할 프레임 수
            video_sampling: 'uniform' 또는 'random'
        """
        self.root = Path(root)
        self.is_train = is_train
        self.video_num_frames = video_num_frames
        self.video_sampling = video_sampling
        
        # transforms_list를 리스트로 정규화
        if transforms_list is None:
            self.transforms_list = [transforms.ToTensor()]
        elif not isinstance(transforms_list, list):
            self.transforms_list = [transforms_list]
        else:
            self.transforms_list = transforms_list
        
        # real (label=0), fake (label=1)로 샘플 수집
        self.samples = []
        self._collect_samples()
        
    def _collect_samples(self):
        """real, fake 폴더에서 파일 수집"""
        real_dir = self.root / 'real'
        fake_dir = self.root / 'fake'
        
        if not real_dir.exists() or not fake_dir.exists():
            raise ValueError(f"Dataset structure requires 'real' and 'fake' folders in {self.root}")
        
        # Real 샘플 수집 (label=0)
        for file_path in real_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS):
                self.samples.append((str(file_path), 0))
        
        # Fake 샘플 수집 (label=1)
        for file_path in fake_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS):
                self.samples.append((str(file_path), 1))
        
        print(f"📊 Loaded {len(self.samples)} samples from {self.root}")
        print(f"   - Real: {sum(1 for _, l in self.samples if l == 0)}")
        print(f"   - Fake: {sum(1 for _, l in self.samples if l == 1)}")
    
    def _load_file(self, path):
        """
        이미지 또는 동영상 로드
        
        Returns:
            이미지: PIL Image
            동영상: List of PIL Images (여러 프레임)
        """
        ext = Path(path).suffix.lower()
        
        if ext in IMAGE_EXTENSIONS:
            return Image.open(path).convert('RGB'), False  # (image, is_video)
        elif ext in VIDEO_EXTENSIONS:
            # 동영상에서 여러 프레임 샘플링
            frames = load_video_frames(
                path, 
                num_frames=self.video_num_frames, 
                sampling=self.video_sampling
            )
            return frames, True  # (frames_list, is_video)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, label = self.samples[index]
        data, is_video = self._load_file(path)
        
        # Single transform인 경우
        if len(self.transforms_list) == 1:
            transform = self.transforms_list[0]
            
            if is_video:
                # 동영상: 각 프레임에 transform 적용
                transformed_frames = [transform(frame) for frame in data]
                return transformed_frames, label, path, True  # (frames, label, path, is_video)
            else:
                # 이미지: 단일 transform
                transformed = transform(data)
                return transformed, label, path, False  # (image, label, path, is_video)
        
        # Multiple transforms인 경우 (Multi-Tower를 위해)
        if is_video:
            # 동영상 + Multi-Tower: 각 프레임에 각 transform 적용
            # [[tf1_frame1, tf1_frame2, ...], [tf2_frame1, tf2_frame2, ...], ...]
            transformed_list = [[tf(frame) for frame in data] for tf in self.transforms_list]
            return transformed_list, label, path, True
        else:
            # 이미지 + Multi-Tower
            transformed_list = [tf(data) for tf in self.transforms_list]
            return transformed_list, label, path, False


class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # ✅ (image, label, path)로 반환
        return img, target, path

def dataset_folder(opt, root):
    """
    데이터셋 생성 팩토리 함수
    
    Args:
        opt: options
        root: 데이터셋 루트 디렉토리
    
    Returns:
        Dataset 인스턴스
    """
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    elif opt.mode == 'filename':
        return FileNameDataset(opt, root)
    elif opt.mode == 'realfake':
        # Multi-Tower 지원
        if hasattr(opt, 'transform_modes') and isinstance(opt.transform_modes, list):
            # Multi-Tower: 여러 transform 적용
            transforms_list = TransformBuilder.build_multi_transforms(
                opt.transform_modes, opt, is_train=opt.isTrain
            )
        else:
            # Single Tower: 단일 transform 적용
            transform_mode = getattr(opt, 'transform_mode', 'texture')
            transforms_list = [TransformBuilder.build_transform(
                transform_mode, opt, is_train=opt.isTrain
            )]
        
        return RealFakeDataset(
            root=root,
            transforms_list=transforms_list,
            is_train=opt.isTrain,
            video_num_frames=getattr(opt, 'video_num_frames', 5),
            video_sampling=getattr(opt, 'video_sampling', 'uniform')
        )
    else:
        raise ValueError(f'opt.mode needs to be binary, filename, or realfake. Got: {opt.mode}')

def texture_transform(img: Image.Image):
    """간단한 텍스처 부각 (Sobel magnitude)"""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag = mag.astype(np.uint8)
    # 다시 PIL.Image로 변환해서 pipeline 이어가기
    mag_rgb = cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(mag_rgb)

def edge_transform(img: Image.Image):
    """간단한 엣지 부각 (Canny edge)"""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    # [H,W] → [H,W,3] (흑백을 3채널로 확장해서 파이프라인 호환)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

def sharpen_transform(img: Image.Image):
    arr = np.array(img)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(arr, -1, kernel)
    return Image.fromarray(sharp)

# def lbp_transform(img: Image.Image, P=8, R=1):
#     gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
#     lbp = local_binary_pattern(gray, P, R, method="uniform")
#     lbp = (lbp / lbp.max() * 255).astype(np.uint8)
#     lbp_rgb = cv2.cvtColor(lbp, cv2.COLOR_GRAY2RGB)
#     return Image.fromarray(lbp_rgb)


class TransformBuilder:
    """
    Multi-Tower 학습을 위한 Transform 빌더
    여러 종류의 transformation을 쉽게 생성하고 관리
    """
    TRANSFORM_TYPES = {
        'texture': texture_transform,
        'edge': edge_transform,
        'sharpen': sharpen_transform,
        # 'lbp': lbp_transform,
    }
    
    @staticmethod
    def build_transform(transform_mode, opt, is_train=True):
        """단일 transform 파이프라인 생성"""
        # Resize
        if not is_train and opt.no_resize:
            rz_func = transforms.Lambda(lambda img: img)
        else:
            rz_func = transforms.Resize((opt.loadSize, opt.loadSize))
        
        # Crop
        if is_train:
            crop_func = transforms.RandomCrop(opt.cropSize)
        elif opt.no_crop:
            crop_func = transforms.Lambda(lambda img: img)
        else:
            crop_func = transforms.CenterCrop(opt.cropSize)
        
        # Flip
        if is_train and not opt.no_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(lambda img: img)
        
        # Transform mode specific
        if transform_mode not in TransformBuilder.TRANSFORM_TYPES:
            raise ValueError(f"Unknown transform_mode: {transform_mode}. "
                           f"Available: {list(TransformBuilder.TRANSFORM_TYPES.keys())}")
        
        additional_transform = TransformBuilder.TRANSFORM_TYPES[transform_mode]
        
        # 전체 파이프라인 구성
        transform_pipeline = transforms.Compose([
            rz_func,
            crop_func,
            flip_func,
            additional_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform_pipeline
    
    @staticmethod
    def build_multi_transforms(transform_modes, opt, is_train=True):
        """
        Multi-Tower를 위한 여러 transform 생성
        
        Args:
            transform_modes: List of transform mode strings (e.g., ['texture', 'edge', 'sharpen'])
            opt: options
            is_train: 학습 모드 여부
            
        Returns:
            List of transform pipelines
        """
        return [TransformBuilder.build_transform(mode, opt, is_train) 
                for mode in transform_modes]

def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        # rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        rz_func = transforms.Resize((opt.loadSize, opt.loadSize))

    if opt.transform_mode == 'texture':
        additional_transform = texture_transform
    elif opt.transform_mode == 'edge':
        additional_transform = edge_transform
    elif opt.transform_mode == 'sharpen':
        additional_transform = sharpen_transform
    # elif opt.transform_mode == 'lbp':
    #     additional_transform = lbp_transform
    else:
        raise NotImplementedError(f'Unknown transform_mode: {opt.transform_mode}')
    
    dset = ImageFolderWithPaths(root,
                                transform=transforms.Compose([
                                    rz_func,
                                    crop_func,
                                    flip_func,
                                    additional_transform,
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406],
                                                         std=[0.229,0.224,0.225])
                                ]))
    dset.transform_mode = opt.transform_mode
    return dset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


# rz_dict = {'bilinear': Image.BILINEAR,
           # 'bicubic': Image.BICUBIC,
           # 'lanczos': Image.LANCZOS,
           # 'nearest': Image.NEAREST}
rz_dict = {'bilinear': InterpolationMode.BILINEAR,
           'bicubic': InterpolationMode.BICUBIC,
           'lanczos': InterpolationMode.LANCZOS,
           'nearest': InterpolationMode.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, (opt.loadSize,opt.loadSize), interpolation=rz_dict[interp])
