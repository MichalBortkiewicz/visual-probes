from somo_utils import load_img, diff, blend
from create_data import _get_in_dirname

import os
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np


# i need some global things, don't yell at me

np.random.seed(42)

IN_DIR = '/local/data/ImageNet'
SUPERPIXELS_DIR = '/local/data/oleszkie/superpixels/'
ALL_LABELS = os.listdir(os.path.join(SUPERPIXELS_DIR, 'train_superpixels'))
MAX_SEARCHED = 50
RESULT_DIR = '/local/data/sieradzk/xai/somo'


print("Building dict")
big_D = {}
for split in ['train', 'val']:
    big_D[split] = defaultdict(list)
    for label in ALL_LABELS:
        label_dir = os.path.join(SUPERPIXELS_DIR, split + '_superpixels', label)
        for sp_filename in filter(lambda x: x.endswith('.png'), os.listdir(label_dir)):
            big_D[split][label].append(os.path.join(label_dir, sp_filename))


def get_patch_path(SP_path):
    sp_filename = SP_path.split('/')[-1]
    patch_filename = "_".join(sp_filename.split('_')[-2:])

    patch_path = '/'.join(SP_path.split('/')[:-1]) + '/' + patch_filename
    patch_path = patch_path.replace('superpixels', 'patches')

    return patch_path


def get_image_name(*, label, sp_file_name):
    if sp_file_name.startswith('val'):
        offset = 17  # len of `val_superpixels_` + 1 for the last underscore
    elif sp_file_name.startswith('train'):
        offset = 19  # len of `train_superpixels_` + 1 for the last underscore
    else:
        raise ValueError(f"Unrecognized superpixel filename: {sp_file_name}")

    if label not in sp_file_name:
        raise ValueError(f"Unrecognized superpixel filename: {sp_file_name}")

    offset += len(label)

    file_name = sp_file_name[offset:-4]  # for .png suffix
    image_filename = file_name.split('_')[0]

    return image_filename


def prepare_result_filename(target_SP_path, source_SP_path):
    target_label = target_SP_path.split('/')[-2]

    target_SP_filename = target_SP_path.split('/')[-1]
    target_patch_filename = "_".join(target_SP_filename.split('_')[-2:])[:-4]

    source_label = source_SP_path.split('/')[-2]
    source_SP_filename = source_SP_path.split('/')[-1]
    source_patch_filename = "_".join(source_SP_filename.split('_')[-2:])[:-4]

    return '-'.join([target_label, target_patch_filename, source_label, source_patch_filename]) + '.png'


def check_if_done(split, target_label, target_SP_path):

    label_dir = os.path.join(RESULT_DIR, split, target_label)
    label_results = os.listdir(label_dir)

    if len(label_results) == 0:
        return False

    target_SP_filename = target_SP_path.split('/')[-1]
    target_patch_filename = "_".join(target_SP_filename.split('_')[-2:])[:-4]

    pattern = target_label + '-' + target_patch_filename

    for result_filename in label_results:
        if result_filename.startswith(pattern):
            return True

    return False


def prepare_tasks():

    imgs_2_SPs = defaultdict(list)

    for split in ['train', 'val']:
        for label in big_D[split].keys():

            IN_label_dir = _get_in_dirname(label)
            IN_ordered_imgs = os.listdir(os.path.join(IN_DIR, split, IN_label_dir))

            for sp_path in big_D[split][label]:
                sp_filename = sp_path.split('/')[-1]
                img_num = get_image_name(label=label, sp_file_name=sp_filename)
                img_path = os.path.join(IN_DIR, split, IN_label_dir, IN_ordered_imgs[int(img_num)])
                imgs_2_SPs[img_path].append(sp_path)

    for img_path, imgs_SPs in imgs_2_SPs.items():
        # TODO: should there be some criterions in choosing target sp?
        imgs_2_SPs[img_path] = np.random.choice(imgs_SPs)

    return list(imgs_2_SPs.items())


def worker(data):

    image_path, target_SP_path = data

    target_label = target_SP_path.split('/')[-2]
    split = target_SP_path.split('/')[-1].split('_')[0]

    if check_if_done(split, target_label, target_SP_path):
        return

    sources = []
    paths = []
    all_sources = []

    for label in filter(lambda x: x != target_label, big_D[split].keys()):
        all_sources += big_D[split][label]

    for sp_path in np.random.choice(all_sources, size=MAX_SEARCHED):
        source = load_img(path=sp_path)
        paths.append(sp_path)
        sources.append(source)

    target_SP_img = load_img(target_SP_path)

    image = load_img(image_path)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)

    sources_arr = np.stack(sources)
    vec_diff = np.vectorize(diff, signature='(224,244,3),(224,224,3)->()')
    scores = vec_diff(target_SP_img, sources_arr)

    source_SP_path = paths[np.argmin(scores)]
    source_patch_path = get_patch_path(source_SP_path)
    source_patch = load_img(source_patch_path)

    target_patch_path = get_patch_path(target_SP_path)
    target_patch = load_img(target_patch_path)

    if len(target_patch.shape) != 3:
        print(f"Skipping due to bad patch dimensions: {target_patch_path}")
        return
    try:
        result = blend(image, target_patch, source_patch)
    except:
        print(f"Skipping image {image_path} with patch {target_patch_path} due to blending error!")
        return

    res_img = Image.fromarray(result.astype(np.uint8))
    res_filename = prepare_result_filename(target_SP_path=target_SP_path, source_SP_path=source_SP_path)
    res_img.save(os.path.join(RESULT_DIR, split, target_label, res_filename))


if __name__ == '__main__':

    print("Preparing tasks")
    tasks = prepare_tasks()

    print("Starting the pool")
    pool = Pool(processes=30)
    for _ in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks)):
        pass


