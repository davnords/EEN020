from argparse import ArgumentParser
from sfm import sfm
import random
import numpy as np

# Todo: 
# - Use RoMa
# - Implement 5 point method
# - Write implementations in CUDA

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=int, required=True, help="Dataset number")
    args = parser.parse_args()

    if args.dataset <1 or args.dataset > 9:
        raise ValueError("Unknown dataset")
    
    # Setting random seeds
    random.seed(1337)
    np.random.seed(1337)
    
    sfm(args.dataset)
