import numpy as np


class CFG:
    # base parameters
    seed = 8620
    deterministic = False
    enable_amp = True
    device = "cuda"
    train_batch_size = 32  # 16
    val_batch_size = 32
    IMG_SIZE = [256, 512]
    # backbone && model parameters
    model_name = "tf_efficientnet_b2"  # tf_efficientnet_b5
    in_channels = 3
    head_dropout = 0.2
    backbone_dropout = 0.2
    backbone_droppath = 0.2
    pretrained_cfg_overlay = (9, 9)  # (3, 3) # (9, 9)
    # optimizer && scheduler parameters
    lr = 2.5e-3  # 5e-3 / 2
    lr_ratio = 5  # 5
    min_lr = 5e-5 / 2
    warmupstep = 0
    # training parameters
    epochs = 15
    es_patience = 4
    # augmentation parameters
    mixup_out_prob = 0.5
    mixup_in_prob1 = 0.5
    mixup_in_prob2 = 0.5
    mixup_alpha_in = 5.0
    mixup_alpha_out = 5.0

    # real mix-up
    # mixup cutmix
    mixup_alpha = 0.5
    cutmix_alpha = 0.5
    switch_prob = 0
    mode = 'batch'

    if_load_pretrained = False
    exp_output_path = './cwt_exp_05'
    chris_data = "eeg_specs_cwt.npy"
    # loss weight
    loss_weight = np.array([1.04866225, 1.10265841, 1.5701029, 3.04291311, 1.55213442, 0.39530419])
    # pretain data path
    pretain_egg_path = 'EEG_Spectrograms_sparcnet_cwt'
    pretrain_model_path = './cwt_pretained_01/best_model_fold0.pth'
    # two stage
    t1 = 10
    t2 = 5
