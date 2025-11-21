# coding: utf-8

import argparse
import random
import numpy as np
import yaml
from train import train_model
import os
import os
import sys



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Propagation Network.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file', type=str, default='../config/AWA2.yaml')
    # parser.add_argument('--config_file', type=str, default='../config/SUN.yaml')
    # parser.add_argument('--config_file', type=str, default='../config/CUB.yaml')
    args = parser.parse_args()

    with open(args.config_file, 'r', encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        print(config)

    config['visual_feat_num'] = 768
    config['trans_feat_num'] = 768
    config['vit_feat'] = True
    config['anchor'] = False
    config['dist_injection'] = True
    config['pred_mode'] = 'fusion'
    config['ablation'] = False
    config['plain_proj'] = False

    train_model(config)