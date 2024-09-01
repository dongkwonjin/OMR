import argparse

def parse_args(cfg):
    root = '/media/dkjin/hdd1'
    parser = argparse.ArgumentParser(description='Hello')
    parser.add_argument('--dataset_dir', default='/home/dkjin/Project/Dataset/VIL-100', help='dataset dir')
    parser.add_argument('--pre_dir', type=str, default=f'{root}/Work/Current/Lane_detection/Project_02/P01_Preprocessing/VIL-100', help='preprocessed data dir')
    args = parser.parse_args()

    cfg = args_to_config(cfg, args)
    return cfg

def args_to_config(cfg, args):
    if args.dataset_dir is not None:
        cfg.dir['dataset'] = args.dataset_dir
        cfg.dir['head_pre'] = args.pre_dir
        cfg.dir['pre0_train'] = cfg.dir['pre0_train'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre0_test'] = cfg.dir['pre0_test'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre1'] = cfg.dir['pre1'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre2'] = cfg.dir['pre2'].replace('--preprocessed data path', args.pre_dir)

    return cfg