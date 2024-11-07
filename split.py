import os
import argparse
import random
from shutil import move
from os import makedirs, path

def split_data(root, env_name, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    source_dir = path.join(root, env_name, 'source')
    train_dir = path.join(root, env_name, 'train')
    val_dir = path.join(root, env_name, 'val')
    test_dir = path.join(root, env_name, 'test')

    makedirs(train_dir, exist_ok=True)
    makedirs(val_dir, exist_ok=True)
    makedirs(test_dir, exist_ok=True)

    video_files = [f for f in os.listdir(source_dir) if f.endswith('.mp4')]
    video_files.sort()  # Ensure consistent ordering

    # Ensure deterministic splitting by setting the random seed
    random.seed(seed)
    random.shuffle(video_files)

    total_files = len(video_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    train_files = video_files[:train_end]
    val_files = video_files[train_end:val_end]
    test_files = video_files[val_end:]

    for file_name in train_files:
        src_path = path.join(source_dir, file_name)
        dst_path = path.join(train_dir, file_name)
        move(src_path, dst_path)

    for file_name in val_files:
        src_path = path.join(source_dir, file_name)
        dst_path = path.join(val_dir, file_name)
        move(src_path, dst_path)

    for file_name in test_files:
        src_path = path.join(source_dir, file_name)
        dst_path = path.join(test_dir, file_name)
        move(src_path, dst_path)

    print(f"Moved {len(train_files)} files to {train_dir}")
    print(f"Moved {len(val_files)} files to {val_dir}")
    print(f"Moved {len(test_files)} files to {test_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split generated videos into train/val/test sets')
    parser.add_argument('--root', type=str, default='./data', help='Root directory of the data')
    parser.add_argument('--env_name', type=str, default='Coinrun', help='Environment name')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of test data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling data')
    args = parser.parse_args()

    split_data(
        root=args.root,
        env_name=args.env_name,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
