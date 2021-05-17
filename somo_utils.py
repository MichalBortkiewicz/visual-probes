import os
from concepts.poisson import process
from PIL import Image
import numpy as np


def get_patch_path(target_path: str) -> str:
    """Returns path to a `patch` images in TCAV results folder structure base in concept path"""
    base_dir = "/".join(target_path.split('/')[:-2])
    concept_dir, file_name = target_path.split('/')[-2:]
    patch_dir = concept_dir + '_patches'
    return os.path.join(base_dir, patch_dir, file_name)


def get_img_path(target_path: str) -> str:
    """Returns path to a full image in TCAV results folder structure base in concept path"""
    base_dir = "/".join(target_path.split('/')[:-2])
    img_ind = str(int(target_path.split('/')[-1].split('_')[1][:-4]) + 1)
    file_name = (4 - len(img_ind)) * '0' + img_ind + '.png'
    return os.path.join(base_dir, 'images', file_name)


def load_img(path: str) -> np.ndarray:
    """Loads an image as numpy array"""
    return np.array(Image.open(path))


def get_concept_inds(img: np.ndarray) -> np.ndarray:
    """Returns a 2D array with rows of index pairs of concepts pixels (non-background)"""
    background = (117, 117, 117)
    return np.stack(np.where(np.all(img != background, axis=-1)), axis=1)


def diff(target: np.ndarray, source: np.ndarray) -> int:
    """Calculates the symmetric difference between two concepts based on overlapping non-background pixels"""
    target_ind = get_concept_inds(target)
    source_ind = get_concept_inds(source)

    nrows, ncols = target_ind.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [target_ind.dtype]}

    target_ind = target_ind.view(dtype)
    source_ind = source_ind.view(dtype)

    diff = len(np.setxor1d(target_ind, source_ind))
    return diff


def find_best_source(target: np.ndarray, source_label: str, data_dir: str, max_concepts=None) -> str:
    """
    Based on a diff function (above) searches for best match in all concepts of a given source label
    :param target: target concepts to change
    :param source_label: label from ImageNet, searches for candidates in its results
    :param data_dir: path to main results directory
    :param max_concepts: maximum number of concepts to search through, default None searches through all
    :return: path to best concept image found
    """

    source_concepts_dir = os.path.join(data_dir, f"{source_label}_4c_explained", "concepts")

    sources = []
    paths = []

    concepts_searched = 0

    for concept_dir in filter(lambda x: not x.endswith('s'), os.listdir(source_concepts_dir)):
        concept_path = os.path.join(source_concepts_dir, concept_dir)
        for img_name in os.listdir(concept_path):
            img_path = os.path.join(concept_path, img_name)
            source = load_img(path=img_path)
            paths.append(img_path)
            sources.append(source)

        concepts_searched += 1

        if max_concepts is not None and concepts_searched == max_concepts:
            break

    sources_arr = np.stack(sources)
    vec_diff = np.vectorize(diff, signature='(224,244,3),(224,224,3)->()')
    scores = vec_diff(target, sources_arr)

    source_path = paths[np.argmin(scores)]

    return source_path


# Poisson blending for Semantic Odd an Out
def blend(image: np.ndarray, target_patch: np.ndarray, source_patch: np.ndarray) -> np.ndarray:
    """Blends source patch into image overwriting the target patch"""

    target_inds = get_concept_inds(target_patch)
    source_inds = get_concept_inds(source_patch)

    target_min = target_inds.min(axis=0)
    target_max = target_inds.max(axis=0)
    target_size = target_max - target_min

    box = np.concatenate([source_inds.min(axis=0)[::-1], source_inds.max(axis=0)[::-1]])
    patch = np.array(Image.fromarray(source_patch).resize(size=target_size[::-1], box=box))

    canvas = 117 * np.ones_like(image)
    canvas[target_min[0]:target_max[0], target_min[1]:target_max[1]] = patch

    mask = np.zeros_like(canvas)[:,:,0]
    mask_inds = get_concept_inds(canvas)
    mask[mask_inds[:,0], mask_inds[:,1]] = 1

    result = np.stack([process(canvas[:,:,i], image[:,:,i], mask) for i in range(3)], axis=-1)
    return np.clip(result, 0, 255)


def semantic_odd_man_out(target_concept_path: str, source_label: str, data_dir: str) -> np.ndarray:
    """Searches for a concept from source_label class to overwrite using Poisson blending"""
    target = load_img(target_concept_path)
    target_patch = load_img(get_patch_path(target_concept_path))
    image = load_img(get_img_path(target_concept_path))

    source_path = find_best_source(target=target, source_label=source_label, data_dir=data_dir)
    source_patch = load_img(get_patch_path(source_path))

    result = blend(image, target_patch, source_patch)

    return result
