import os
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
from torchvision.datasets import MNIST
from torchvision import transforms


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_mnist(root: str, train: bool = True):
    """
    Download and load MNIST images.

    Returns:
        images: numpy array with shape [N, 28, 28], dtype uint8
    """
    dataset = MNIST(
        root=root,
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )

    images = []
    for img, _ in dataset:
        # img: [1, 28, 28], float tensor in [0, 1]
        img = (img.squeeze(0).numpy() * 255.0).astype(np.uint8)
        images.append(img)

    images = np.stack(images, axis=0)
    return images


def sample_velocity(min_speed: float, max_speed: float):
    """
    Sample a 2D velocity vector.

    We avoid very small velocities by sampling speed magnitude
    and random direction.
    """
    speed = np.random.uniform(min_speed, max_speed)
    angle = np.random.uniform(0, 2 * np.pi)

    vx = speed * np.cos(angle)
    vy = speed * np.sin(angle)

    return vx, vy


def paste_digit(canvas: np.ndarray, digit: np.ndarray, x: int, y: int):
    """
    Paste digit onto canvas using max composition.

    Args:
        canvas: [canvas_size, canvas_size], uint8
        digit: [digit_size, digit_size], uint8
        x: top-left x coordinate
        y: top-left y coordinate
    """
    h, w = digit.shape
    canvas_patch = canvas[y : y + h, x : x + w]

    np.maximum(canvas_patch, digit, out=canvas_patch)


def generate_one_sequence(
    mnist_images: np.ndarray,
    seq_len: int = 20,
    canvas_size: int = 64,
    digit_size: int = 28,
    num_digits: int = 1,
    min_speed: float = 2.0,
    max_speed: float = 4.0,
):
    """
    Generate one Moving MNIST sequence.

    Returns:
        video: [seq_len, canvas_size, canvas_size], uint8
    """
    video = np.zeros((seq_len, canvas_size, canvas_size), dtype=np.uint8)

    digits = []
    positions = []
    velocities = []

    max_pos = canvas_size - digit_size

    for _ in range(num_digits):
        idx = np.random.randint(0, len(mnist_images))
        digit = mnist_images[idx]

        x = np.random.uniform(0, max_pos)
        y = np.random.uniform(0, max_pos)

        vx, vy = sample_velocity(min_speed, max_speed)

        digits.append(digit)
        positions.append([x, y])
        velocities.append([vx, vy])

    for t in range(seq_len):
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        for i in range(num_digits):
            digit = digits[i]
            x, y = positions[i]
            vx, vy = velocities[i]

            # integer position for rendering
            x_int = int(round(x))
            y_int = int(round(y))

            x_int = np.clip(x_int, 0, max_pos)
            y_int = np.clip(y_int, 0, max_pos)

            paste_digit(canvas, digit, x_int, y_int)

            # update position
            x_next = x + vx
            y_next = y + vy

            # bounce on x boundary
            if x_next < 0:
                x_next = -x_next
                vx = -vx
            elif x_next > max_pos:
                x_next = 2 * max_pos - x_next
                vx = -vx

            # bounce on y boundary
            if y_next < 0:
                y_next = -y_next
                vy = -vy
            elif y_next > max_pos:
                y_next = 2 * max_pos - y_next
                vy = -vy

            positions[i] = [x_next, y_next]
            velocities[i] = [vx, vy]

        video[t] = canvas

    return video


def generate_dataset(
    mnist_images: np.ndarray,
    num_sequences: int,
    seq_len: int,
    canvas_size: int,
    digit_size: int,
    num_digits: int,
    min_speed: float,
    max_speed: float,
):
    """
    Generate full Moving MNIST dataset.

    Returns:
        data: [num_sequences, seq_len, canvas_size, canvas_size], uint8
    """
    data = np.zeros(
        (num_sequences, seq_len, canvas_size, canvas_size),
        dtype=np.uint8,
    )

    for i in tqdm(range(num_sequences), desc="Generating Moving MNIST"):
        data[i] = generate_one_sequence(
            mnist_images=mnist_images,
            seq_len=seq_len,
            canvas_size=canvas_size,
            digit_size=digit_size,
            num_digits=num_digits,
            min_speed=min_speed,
            max_speed=max_speed,
        )

    return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mnist_root", type=str, default="data/mnist")
    parser.add_argument("--output_dir", type=str, default="data/moving_mnist")

    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_test", type=int, default=1000)

    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--canvas_size", type=int, default=64)
    parser.add_argument("--digit_size", type=int, default=28)
    parser.add_argument("--num_digits", type=int, default=1)

    parser.add_argument("--min_speed", type=float, default=2.0)
    parser.add_argument("--max_speed", type=float, default=4.0)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading MNIST training set...")
    mnist_train = load_mnist(args.mnist_root, train=True)

    print("Loading MNIST test set...")
    mnist_test = load_mnist(args.mnist_root, train=False)

    print("Generating training Moving MNIST...")
    train_data = generate_dataset(
        mnist_images=mnist_train,
        num_sequences=args.num_train,
        seq_len=args.seq_len,
        canvas_size=args.canvas_size,
        digit_size=args.digit_size,
        num_digits=args.num_digits,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
    )

    print("Generating test Moving MNIST...")
    test_data = generate_dataset(
        mnist_images=mnist_test,
        num_sequences=args.num_test,
        seq_len=args.seq_len,
        canvas_size=args.canvas_size,
        digit_size=args.digit_size,
        num_digits=args.num_digits,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
    )

    train_path = os.path.join(args.output_dir, "train.npy")
    test_path = os.path.join(args.output_dir, "test.npy")

    np.save(train_path, train_data)
    np.save(test_path, test_data)

    print("Done.")
    print(f"Saved train data to: {train_path}, shape={train_data.shape}, dtype={train_data.dtype}")
    print(f"Saved test data to : {test_path}, shape={test_data.shape}, dtype={test_data.dtype}")


if __name__ == "__main__":
    main()