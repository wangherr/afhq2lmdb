import argparse
import os

import lmdb
import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm


def main(input_root, out_root):
    for mode_name in ['train', 'val']:
        img_dir = os.path.join(input_root, mode_name)
        dataset = torchvision.datasets.ImageFolder(img_dir)
        print(f'[info] mode: {mode_name} \t class~idx: {dataset.class_to_idx}')

        for kind_name in ['cat', 'dog', 'wild']:
            out_dir = os.path.join(out_root, f'{kind_name}', mode_name+'.lmdb')
            os.makedirs(out_dir, exist_ok=True)
            with lmdb.open(out_dir, map_size=10*1024**2) as env:
                count = prepare(env, dataset, kind_name)
            print(f'[info] count: {count}, save {kind_name}-{mode_name} in {out_dir}')
            print('-------------------------------------------------')


def prepare(env, dataset, kind_name):
    count = 0
    for img_path, label in tqdm(dataset.imgs):
        if dataset.class_to_idx[kind_name] != label:
            continue

        img = Image.open(img_path)
        img = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[1], img.size[0], 3)

        with env.begin(write=True) as txn:
            txn.put(str(count).encode(), img)

        count += 1
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser('create afhq lmdb')
    parser.add_argument('--input_root', type=str, default="afhq",
                        help='dataset dir')
    parser.add_argument('--out_root', type=str, default="afhq_lmdb",
                        help='dir to save lmdb dataset')
    args = parser.parse_args()
    main(args.input_root, args.out_root)
