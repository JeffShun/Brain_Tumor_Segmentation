import sys, os
work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(work_dir)
from custom.model.backbones.Cascaded_Unet import *
from custom.model.model_head import *
from custom.model.model_network import *
from custom.utils.common_tools import *
from custom.dataset.dataset import MyDataset

class network_cfg:
    # img
    crop_box = [15, 143, 24, 216, 40, 200]
    win_clip = None

    # network
    network = Model_Network(
        backbone = Cascaded_Unet(indim=4, base_channel=12),
        head = Model_Head(in_channel1=12, in_channel2=12, head1=1, head2=2, label_smooth=0.0),
        apply_sync_batchnorm=False,
    )

    # dataset
    train_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/train.lst",
        crop_box = crop_box,
        transforms = Compose([
            to_tensor(),
            normlize(win_clip=win_clip),
            random_flip(prob=0.5),
            random_gamma_transform(gamma_range=[0.8,1.2], prob=0.5)
            ])
        )
    valid_dataset = MyDataset(
        dst_list_file = work_dir + "/train_data/processed_data/valid.lst",
        crop_box = crop_box,
        transforms = Compose([
            to_tensor(),           
            normlize(win_clip=win_clip)
            ])
        )
    
    # dataloader
    batchsize = 2
    shuffle = True
    num_workers = 8
    drop_last = False

    # optimizer
    lr = 1e-4
    weight_decay = 1e-4

    # scheduler
    milestones = [50,90]
    gamma = 0.1
    warmup_factor = 0.1
    warmup_iters = 50
    warmup_method = "linear"
    last_epoch = -1

    # debug
    valid_interval = 5
    log_dir = work_dir + "/Logs"
    checkpoints_dir = work_dir + '/checkpoints/v1'
    checkpoint_save_interval = 2
    total_epochs = 200
    load_from = ''

    # others
    device = 'cuda'
    dist_backend = 'nccl'
    dist_url = 'env://'
