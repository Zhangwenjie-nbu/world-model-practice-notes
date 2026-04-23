import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/moving_mnist/train.npy")
    args = parser.parse_args()

    data = np.load(args.path)

    print("Path :", args.path)
    print("Shape:", data.shape)
    print("Dtype:", data.dtype)
    print("Min  :", data.min())
    print("Max  :", data.max())


if __name__ == "__main__":
    main()