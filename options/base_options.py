import argparse
import os
import time
import util
import torch
#import models
#import data
 

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--mode', default='binary')
        parser.add_argument('--arch', type=str, default='res50', help='architecture for binary classification')

        # data augmentation
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--blur_prob', type=float, default=0)
        parser.add_argument('--blur_sig', default='0.5')
        parser.add_argument('--jpg_prob', type=float, default=0)
        parser.add_argument('--jpg_method', default='cv2')
        parser.add_argument('--jpg_qual', default='75')

        parser.add_argument('--dataroot', default='./dataset/', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--classes', default='', help='image classes to train on')
        parser.add_argument('--class_bal', action='store_true')
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=224, help='then crop to this size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--resize_or_crop', type=str, default='scale_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        parser.add_argument('--delr_freq', type=int, default=20, help='frequency of changing lr')

        # Tower 모드 선택
        parser.add_argument('--tower_mode', type=str, default='single', choices=['single', 'multi'],
                          help='Tower mode: single or multi (multi is not implemented yet)')
        
        # Transform 관련 옵션
        parser.add_argument('--transform_mode', type=str, default='texture', 
                          choices=['texture', 'edge', 'sharpen'],
                          help='Single-Tower transform mode: texture, edge, sharpen')
        parser.add_argument('--transform_modes', type=str, default=None,
                          help='Multi-Tower transform modes (comma-separated): texture,edge,sharpen')
        
        # Multi-Tower 관련 옵션 (TODO: Not implemented yet)
        parser.add_argument('--fusion_method', type=str, default='concat',
                          choices=['concat', 'average', 'attention'],
                          help='Fusion method for Multi-Tower: concat, average, attention (NOT IMPLEMENTED)')
        # Video 관련 옵션
        parser.add_argument('--video_num_frames', type=int, default=5,
                          help='Number of frames to sample from video')
        parser.add_argument('--video_sampling', type=str, default='uniform',
                          choices=['uniform', 'random'],
                          help='Video frame sampling strategy: uniform or random')
        parser.add_argument('--video_voting', type=str, default='max',
                          choices=['max', 'avg', 'majority'],
                          help='Video prediction voting: max (max prob), avg (average prob), majority (most common)')
        
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return opt #parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        opt.name = opt.name + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # additional
        opt.classes = opt.classes.split(',')
        opt.rz_interp = opt.rz_interp.split(',')
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        opt.jpg_method = opt.jpg_method.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")
        
        # Multi-Tower: transform_modes 파싱
        if opt.transform_modes is not None and opt.transform_modes != '':
            opt.transform_modes = [mode.strip() for mode in opt.transform_modes.split(',')]
            print(f"🔀 Multi-Tower mode enabled with transforms: {opt.transform_modes}")
        else:
            opt.transform_modes = None

        self.opt = opt
        return self.opt
