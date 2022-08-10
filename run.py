# -*- coding: utf-8 -*-
# @Author: Ananth Ravi Kumar
# @Date:   2020-06-08 15:59:07
# @Last Modified by:   Ananth
# @Last Modified time: 2020-10-21 23:37:16


from src import run_experiment 
from src.utils import CIFARLoader
from argparse import ArgumentParser
 

def select_reader(data_configs):
    dataset = data_configs.get("dataset")
    if dataset == 'cifar10':
        return CIFARLoader(data_configs) 
    else: 
        raise NotImplementedError("Dataset not currently supported")


def parse_args():
    parser = ArgumentParser(description='self-supervised experiment details')
    parser.add_argument('--data_config', required=True, type='str', help="dataset configuration options")
    parser.add_argument('--model_config', required=True, type='str', help='network configuration options')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args() 

    reader = select_reader(args.data_config)    
    all_results = run_experiment(reader, args.model_config)

