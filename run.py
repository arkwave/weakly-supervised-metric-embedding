# -*- coding: utf-8 -*-
# @Author: Ananth Ravi Kumar
# @Date:   2020-06-08 15:59:07
# @Last Modified by:   Ananth
# @Last Modified time: 2020-10-21 23:37:16


from src import run_experiment 

all_results = run_experiment("pairwise", "vae", "cifar10", [0.75], [3], 
                             num_iters=10, batch_size=32, normalize=True,
                             feature_extractor='hog')
