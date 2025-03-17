class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = 'E:/project/trust_fusion/MFJA'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = 'E:/project/trust_fusion/MFJA/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = 'E:/project/trust_fusion/MFJA/pretrained_networks'
        self.got10k_val_dir = 'E:/dataset/got10k/val'
        self.lasot_lmdb_dir = 'E:/dataset/lasot_lmdb'
        self.got10k_lmdb_dir = 'E:/dataset/got10k_lmdb'
        self.trackingnet_lmdb_dir = 'E:/dataset/trackingnet_lmdb'
        self.coco_lmdb_dir = 'E:/dataset/coco_lmdb'
        self.coco_dir = 'E:/dataset/coco'
        self.lasot_dir = 'E:/dataset/lasot'
        self.got10k_dir = 'E:/dataset/got10k/train'
        self.trackingnet_dir = 'E:/dataset/trackingnet'
        self.depthtrack_dir = 'E:/dataset/depthtrack/train'
        self.lasher_dir = 'E:/dataset/lasher/trainingset'
        self.visevent_dir = 'E:/dataset/visevent/train'
