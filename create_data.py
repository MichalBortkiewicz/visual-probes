import sys
import os
import shutil
import numpy as np
import socket

from config import ROOT_DIR

TMP_IN_TARGET_FOLDER = "train"
IMAGE_NET_PATH = "//home/mbortkie/cl_probing/continual-probing/data/ImageNet"
DATA_DIR = "//home/mbortkie/cl_probing/continual-probing/data/"


def _get_in_dirname(target_class: str) -> str:

    with open(os.path.join(ROOT_DIR, "visual_probes", "imagenet_dirs.txt"), "r") as f:
        lines = f.readlines()

    class_to_dir = {}
    for line in lines:
        dir_name, _, cls = line.strip().split()
        class_to_dir[cls] = dir_name

    if target_class not in class_to_dir:
        raise ValueError(
            f"Target class {target_class} not found in imagenet_dirs.txt! Make sure you spelled correctly."
        )
    else:
        return class_to_dir[target_class]


if __name__ == "__main__":

    # TODO: make those parameters?
    n_exps = 10
    n_imgs = 50

    if socket.gethostname() != "dgx2":
        print(
            "WARNING: You seem to be running this not on dgx2.gmum, please set proper IMAGE_NET_PATH in this file!"
        )

    assert len(sys.argv) == 2, "Please pass only one argument"
    target_class = sys.argv[1]
    DATA_DIR = os.path.join(DATA_DIR, target_class)
    target_dirname = _get_in_dirname(target_class)

    # 1. copy target class
    target_path = os.path.join(DATA_DIR, target_class)
    in_target = os.path.join(IMAGE_NET_PATH, TMP_IN_TARGET_FOLDER, target_dirname)

    os.makedirs(os.path.join(target_path), exist_ok=True)

    for zebra_file in filter(lambda x: x.endswith(".JPEG"), os.listdir(in_target)):
        shutil.copyfile(
            src=os.path.join(in_target, zebra_file),
            dst=os.path.join(target_path, zebra_file),
        )

    # 2. copy to random_discovery
    random_path = os.path.join(DATA_DIR, "random_discovery")
    os.makedirs(os.path.join(random_path), exist_ok=True)

    val_dirs = os.listdir(os.path.join(IMAGE_NET_PATH, "val"))

    for i in range(n_imgs):
        rand_dir = np.random.choice(val_dirs)
        rand_file = np.random.choice(
            os.listdir(os.path.join(IMAGE_NET_PATH, "val", rand_dir))
        )

        shutil.copyfile(
            src=os.path.join(IMAGE_NET_PATH, "val", rand_dir, rand_file),
            dst=os.path.join(random_path, rand_file),
        )

    # 3. copy to random_500_X
    val_dirs = os.listdir(os.path.join(IMAGE_NET_PATH, "val"))

    for exp_id in range(n_exps):

        exp_path = os.path.join(DATA_DIR, f"random500_{exp_id}")
        os.makedirs(os.path.join(exp_path), exist_ok=True)

        for i in range(n_imgs):
            rand_dir = np.random.choice(val_dirs)
            rand_file = np.random.choice(
                os.listdir(os.path.join(IMAGE_NET_PATH, "val", rand_dir))
            )

            shutil.copyfile(
                src=os.path.join(IMAGE_NET_PATH, "val", rand_dir, rand_file),
                dst=os.path.join(exp_path, rand_file),
            )
