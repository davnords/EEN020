from argparse import ArgumentParser
from sfm import sfm
import random
import numpy as np
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=int, required=True, help="Dataset number")
    parser.add_argument("--plots", type=str, default='partial', help="Plotting option. Choose between 'full' and 'partial'")
    args = parser.parse_args()

    if args.dataset <1 or args.dataset > 11:
        raise ValueError("Unknown dataset")
    
    # Setting random seeds
    random.seed(1337)
    np.random.seed(1337)
    
    os.makedirs(f"./storage", exist_ok=True)
    os.makedirs(f"./plots", exist_ok=True)
    
    sfm(args)
