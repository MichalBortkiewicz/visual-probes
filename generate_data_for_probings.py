import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


from PIL import Image
from create_data import _get_in_dirname
import sys

sys.path.append("./ACE")
from ACE.ace_helpers import load_image_from_file
from ACE import ace_helpers
from ACE.ace import ConceptDiscovery
from tcav import utils
from pathlib import Path

# import tensorflow.compat.v1 as tf
#
# tf.disable_eager_execution()
# import tensorflow_hub as hub

# from simclr_pytorch import EncodeProject

IMAGE_NET_PATH = "/local/data/ImageNet"
ROOT_DIR = "/home/mbortkie/"

top_concepts = pd.read_csv(
    os.path.join(ROOT_DIR, "visual_probes", "top_concepts_df.csv")
)["concept"].tolist()

CLASSES = [
    "zebra",
    "bison",
    "koala",
    "jaguar",
    "chimpanzee",
    "hog",
    "hamster",
    "lion",
    "beaver",
    "lynx",
    "sports_car",
    "airliner",
    "jeep",
    "passenger_car",
    "steam_locomotive",
    "cab",
    "garbage_truck",
    "warplane",
    "ambulance",
    "police_van",
    "planetarium",
    "castle",
    "church",
    "mosque",
    "triumphal_arch",
    "barn",
    "stupa",
    "suspension_bridge",
    "steel_arch_bridge",
    "viaduct",
    "sax",
    "flute",
    "cornet",
    "panpipe",
    "cello",
    "acoustic_guitar",
    "grand_piano",
    "banjo",
    "maraca",
    "chime",
    "fig",
    "custard_apple",
    "banana",
    "corn",
    "lemon",
    "pomegranate",
    "pineapple",
    "jackfruit",
    "strawberry",
    "orange",
]
test_classes = CLASSES

assert test_classes.__len__() == 50


class ImagenetDataset(Dataset):
    def __init__(self, root_path: str, phase: str, target_class: str, transforms=None):
        self.root_path = root_path
        self.phase = phase
        self.target_class = target_class
        self.transforms = transforms
        self.target_dirname = _get_in_dirname(self.target_class)
        self.directory = os.path.join(self.root_path, self.phase, self.target_dirname)
        filenames = [
            filename
            for filename in filter(
                lambda x: x.endswith(".JPEG"), os.listdir(self.directory)
            )
        ]
        self.filenames = [
            os.path.join(self.directory, filename) for filename in filenames
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.filenames[idx]
        img = load_image_from_file(img_name, (224, 224))

        # convert 1ch/4ch images to 3ch images
        if img is None:
            img = np.array(Image.open(img_name).resize((224, 224), Image.BILINEAR))
            img = np.float32(img) / 255.0
            if len(img.shape) == 2:
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            elif len(img.shape) == 3:
                if img.shape[2] == 4:
                    img = img[:, :, :3]

        # apply transforms
        if self.transforms:
            img = self.transforms(img)

        return img


class PatchesDataset(Dataset):
    def __init__(self, path: str, phase: str, target_class: str, transforms=None):
        self.path = os.path.join(path, f"{phase}_superpixels")
        self.target_class = target_class
        self.transforms = transforms
        self.filenames = []
        for file_name in sorted(os.listdir(os.path.join(self.path, target_class))):
            if file_name.endswith(".png"):
                # Use only 20 first image for patch generation
                if int(file_name.split("_")[-2]) < 20:
                    self.filenames.append(
                        os.path.join(self.path, target_class, file_name)
                    )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.filenames[idx]
        img = load_image_from_file(img_name, (224, 224))

        # convert 1ch/4ch images to 3ch images
        if img is None:
            img = np.array(Image.open(img_name).resize((224, 224), Image.BILINEAR))
            img = np.float32(img) / 255.0
            if len(img.shape) == 2:
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            elif len(img.shape) == 3:
                if img.shape[2] == 4:
                    img = img[:, :, :3]
        # apply transforms
        if self.transforms:
            img = self.transforms(img)
        return img


class SomoDataset(Dataset):
    def __init__(self, path: str, phase: str, target_class: str, transforms=None):
        self.path = path
        self.target_class = target_class
        self.transforms = transforms
        self.filenames = []
        for file_name in sorted(
            os.listdir(os.path.join(self.path, phase, target_class))
        ):
            if file_name.endswith(".png"):
                self.filenames.append(
                    os.path.join(self.path, phase, target_class, file_name)
                )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.filenames[idx]
        img = load_image_from_file(img_name, (224, 224))

        # convert 1ch/4ch images to 3ch images
        if img is None:
            img = np.array(Image.open(img_name).resize((224, 224), Image.BILINEAR))
            img = np.float32(img) / 255.0
            if len(img.shape) == 2:
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            elif len(img.shape) == 3:
                if img.shape[2] == 4:
                    img = img[:, :, :3]
        # apply transforms
        if self.transforms:
            img = self.transforms(img)
        return img


class ResNet(torch.nn.Module):
    def __init__(self, net_name, pretrained=False, use_fc=False):
        super().__init__()
        base_model = models.__dict__[net_name](pretrained=pretrained)
        self.encoder = torch.nn.Sequential(*list(base_model.children())[:-1])

        self.use_fc = use_fc
        if self.use_fc:
            self.fc = torch.nn.Linear(2048, 512)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        if self.use_fc:
            x = self.fc(x)
        return x


def calc_image_activations(
    phase: str,
    target_class: str,
    bottlenecks,
    image_net_path=os.path.join(ROOT_DIR, "data", "ImageNet"),
    model_to_run="GoogleNet",
    model_path=os.path.join(
        ROOT_DIR, "visual_probes", "ACE", "tensorflow_inception_graph.pb"
    ),
    labels_path=os.path.join(ROOT_DIR, "visual_probes", "ACE", "imagenet_labels.txt"),
):

    # Make tensorflow model
    sess = utils.create_session()
    model = ace_helpers.make_model(sess, model_to_run, model_path, labels_path)

    # Create dummy ConceptDiscovery object
    # We will need it to generate image activations
    dummy_cd = ConceptDiscovery(
        model=model,
        target_class=None,
        random_concept=None,
        bottlenecks=bottlenecks,
        sess=None,
        source_dir=None,
        activation_dir=None,
        cav_dir=None,
        num_workers=16,
    )

    # Create Dataset object
    ds = ImagenetDataset(image_net_path, phase, target_class)

    # Save activations
    activations = []

    # Iterate through images
    print(f"Target class: {target_class}, phase: {phase}")
    for idx, image in enumerate(ds):
        print(f"Image idx: {idx} out of {len(ds)}")

        # Extract superpixels embeddings
        sp_outputs = dummy_cd._return_superpixels(
            image, "slic", param_dict={"n_segments": [15, 50, 80]}
        )
        image_superpixels, image_patches = sp_outputs
        superpixels_embedding = ace_helpers.get_acts_from_images(
            image_superpixels, model, dummy_cd.bottlenecks[0]
        )
        superpixels_embedding = np.mean(superpixels_embedding, axis=(1, 2))
        activations.append(superpixels_embedding)

    return activations


def calc_patches(
    phase: str,
    target_class: str,
    bottlenecks,
    image_net_path="/local/data/ImageNet",
    model_to_run="GoogleNet",
    model_path="ACE/tensorflow_inception_graph.pb",
    labels_path="ACE/imagenet_labels.txt",
    path="/local/data/oleszkie",
):

    # Make tensorflow model
    sess = utils.create_session()
    model = ace_helpers.make_model(sess, model_to_run, model_path, labels_path)

    # Create dummy ConceptDiscovery object
    # We will need it to generate image activations
    dummy_cd = ConceptDiscovery(
        model=model,
        target_class=None,
        random_concept=None,
        bottlenecks=bottlenecks,
        sess=None,
        source_dir=None,
        activation_dir=None,
        cav_dir=None,
        num_workers=16,
    )

    # Create Dataset object
    ds = ImagenetDataset(image_net_path, phase, target_class)

    # Iterate through images
    for idx, image in enumerate(ds):

        # Extract superpixels embeddings
        sp_outputs = dummy_cd._return_superpixels(
            image, "slic", param_dict={"n_segments": [15, 50, 80]}
        )
        image_superpixels, image_patches = sp_outputs
        for sp_idx, superpixel in enumerate(image_superpixels):
            superpixel = (np.clip(superpixel, 0, 1) * 256).astype(np.uint8)
            with open(
                os.path.join(
                    path,
                    f"/superpixels/{phase}_superpixels/{target_class}/{idx:06}_{sp_idx:06}.png",
                    "wb",
                )
            ) as sp_file:
                Image.fromarray(superpixel).save(sp_file, format="PNG")
        for patch_idx, patch in enumerate(image_patches):
            patch = (np.clip(patch, 0, 1) * 256).astype(np.uint8)
            with open(
                os.path.join(
                    path,
                    f"/patches/{phase}_patches/{target_class}/{idx:06}_{patch_idx:06}.png",
                    "wb",
                )
            ) as patch_file:
                Image.fromarray(patch).save(patch_file, format="PNG")
    return


#
# def create_ss_patch_embeddings(
#     path, phase, target_class, representation, local_path="/local/data/oleszkie"
# ):
#
#     # Load self-supervised model
#     model = None
#     sess = None
#
#     if representation == "swav":
#         model = torch.hub.load("facebookresearch/swav", "resnet50")
#         # We don't need output layer
#         model = torch.nn.Sequential(*list(model.children())[:-1])
#
#     elif representation == "simclr":
#         model = hub.Module(
#             "gs://simclr-checkpoints/simclrv2/pretrained/r50_1x_sk0/hub/",
#             trainable=False,
#         )
#
#     elif representation == "byol":
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         model = ResNet("resnet50", pretrained=False, use_fc=False).to(device)
#
#         # load encoder
#         model_path = os.path.join(
#             local_path, "models/resnet50_byol_imagenet2012.pth.tar"
#         )
#         checkpoint = torch.load(model_path, map_location=device)["online_backbone"]
#         state_dict = {}
#         length = len(model.encoder.state_dict())
#         for name, param in zip(
#             model.encoder.state_dict(), list(checkpoint.values())[:length]
#         ):
#             state_dict[name] = param
#         model.encoder.load_state_dict(state_dict, strict=True)
#         model.eval()
#
#     elif representation == "moco":
#
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         model = models.__dict__["resnet50"]().to(device)
#
#         # load pretrained weights from the file
#         model_path = os.path.join(local_path, "models/moco_v1_200ep_pretrain.pth.tar")
#         checkpoint = torch.load(model_path, map_location=device)
#         state_dict = checkpoint["state_dict"]
#
#         # preprocess state dict
#         for key in list(state_dict.keys()):
#             if key.startswith("module.encoder_q") and not key.startswith(
#                 "module.encoder_q.fc"
#             ):
#                 state_dict[key[len("module.encoder_q.") :]] = state_dict[key]
#             del state_dict[key]
#
#         # load model
#         msg = model.load_state_dict(state_dict, strict=False)
#         assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
#
#         # We don't need output layer
#         model = torch.nn.Sequential(*list(model.children())[:-1])
#
#         model.eval()
#
#     model = model.cuda()
#
#     if representation in ["swav", "byol", "moco"]:
#         eval_model = lambda img: model(img).detach().cpu().numpy().squeeze()
#
#     # Calulcate self-supervised embeddings
#
#     ds = PatchesDataset(
#         path, phase, target_class, transforms.Compose([transforms.ToTensor()])
#     )
#     # ds = SomoDataset(path, phase, target_class, transforms.Compose([transforms.ToTensor()]))
#     loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=20)
#
#     result = None
#     for idx, image in enumerate(loader):
#         image = image.cuda()
#         if result is None:
#             result = eval_model(image)
#         else:
#             batch_result = eval_model(image)
#             result = np.vstack((result, batch_result))
#
#     if sess:
#         sess.close()
#
#     # Serialzie
#     save_path = os.path.join(
#         path, f"{phase}_superpixels", target_class, f"{representation}_{cut_layers}"
#     )
#     np.save(save_path, result)
#     print(f"Results {result.shape} saved at {save_path}")
#
#
# def create_ss_embeddings(
#     representation: str, phase: str, target_classes, local_path="/local/data/oleszkie"
# ):
#
#     # Load self-supervised model
#
#     model = None
#     sess = None
#
#     if representation == "swav":
#         model = torch.hub.load("facebookresearch/swav", "resnet50")
#         # We don't need output layer
#         model = torch.nn.Sequential(*list(model.children())[:-1])
#
#     elif representation == "simclr":
#         model = hub.Module(
#             "gs://simclr-checkpoints/simclrv2/pretrained/r50_1x_sk0/hub/",
#             trainable=False,
#         )
#
#     elif representation == "byol":
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         model = ResNet("resnet50", pretrained=False, use_fc=False).to(device)
#
#         # load encoder
#         model_path = os.path.join(
#             local_path, "models/resnet50_byol_imagenet2012.pth.tar"
#         )
#         checkpoint = torch.load(model_path, map_location=device)["online_backbone"]
#         state_dict = {}
#         length = len(model.encoder.state_dict())
#         for name, param in zip(
#             model.encoder.state_dict(), list(checkpoint.values())[:length]
#         ):
#             state_dict[name] = param
#         model.encoder.load_state_dict(state_dict, strict=True)
#         model.eval()
#
#     elif representation == "moco":
#
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         model = models.__dict__["resnet50"]().to(device)
#
#         # load pretrained weights from the file
#         model_path = os.path.join(local_path, "/models/moco_v1_200ep_pretrain.pth.tar")
#         checkpoint = torch.load(model_path, map_location=device)
#         state_dict = checkpoint["state_dict"]
#
#         # preprocess state dict
#         for key in list(state_dict.keys()):
#             if key.startswith("module.encoder_q") and not key.startswith(
#                 "module.encoder_q.fc"
#             ):
#                 state_dict[key[len("module.encoder_q.") :]] = state_dict[key]
#             del state_dict[key]
#
#         # load model
#         msg = model.load_state_dict(state_dict, strict=False)
#         assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
#
#         # We don't need output layer
#         model = torch.nn.Sequential(*list(model.children())[:-1])
#
#         model.eval()
#
#     model = model.cuda()
#
#     for target_class in target_classes:
#
#         if representation in ["swav", "byol", "moco"]:
#             eval_model = lambda img: model(img).detach().cpu().numpy().squeeze()
#
#         elif representation == "simclr":
#             sess = tf.Session()
#             sess.run(tf.global_variables_initializer())
#             eval_model = lambda img: model(np.moveaxis(img.cpu().numpy(), 1, 3)).eval(
#                 session=sess
#             )
#
#         # Create dataset object
#         ds = ImagenetDataset(
#             IMAGE_NET_PATH,
#             phase,
#             target_class,
#             transforms.Compose([transforms.ToTensor()]),
#         )
#         loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=16)
#
#         # Calulcate self-supervised embeddings
#         result = None
#         for idx, image in enumerate(loader):
#             image = image.cuda()
#             print(target_class + " " + str(idx))
#             if result is None:
#                 result = eval_model(image)
#             else:
#                 result = np.vstack((result, eval_model(image)))
#             print(result.shape)
#
#         # Serialzie
#         with open(
#             os.path.join(
#                 local_path,
#                 "embeddings/{phase}_embd_{representation}_55_{target_class}.pkl",
#             ),
#             "wb",
#         ) as file:
#             pickle.dump(result, file)
#         if sess:
#             sess.close()
#         print(
#             f"Saved {result.shape} at {local_path}/embeddings/{phase}_embd_{representation}_55_{target_class}.pkl"
#         )
#     return
#


def create_concept_labels(
    phase: str,
    target_class: str,
    all_classes: List[str],
    top_concepts: List[str],
    local_path="/local/data/oleszkie",
):

    # Generate filenames with concept centers
    centers_pkls = [
        os.path.join(local_path, "concepts", f"{clas}_dict.pkl") for clas in all_classes
    ]

    # Get dict of (concept_name, concept_center)
    centers = {}

    # Read file with concept centers and radiuses
    print(f"Target class: {target_class}, phase: {phase}, getting concepts centers")
    for idx, centers_pkl in enumerate(centers_pkls):
        print(f"Pickle file idx: {idx}, out of {len(centers_pkls)}")
        with open(centers_pkl, "rb") as centers_file:
            centers_pkl = pickle.load(centers_file)

            # Parse file with concept centers and radiuses
            for concept_name in centers_pkl["concepts"]:
                if concept_name in top_concepts:
                    center = centers_pkl[concept_name + "_" + "center"]
                    centers[concept_name] = center

    # Convert dict to pandas dataframe
    df_centers = pd.DataFrame.from_dict(centers)

    # List of final labels
    labels = []

    # Iterate through activations
    activations = None
    with open(
        os.path.join(
            local_path, "activations", f"{phase}_activations_{target_class}.pkl"
        ),
        "rb",
    ) as activations_file:
        activations = pickle.load(activations_file)

    print(f"Target class: {target_class}, phase: {phase}, getting activations")
    for idx, image in enumerate(activations):
        print(f"Activations ile idx: {idx}, out of {len(activations)}")
        # Initialize is concept present dict
        is_concept_present = {concept: 0 for concept in df_centers.columns}

        # Iterate through all superpixels and find concept for each
        for superpixel in image:

            # Calculate distances between superpixel and concepts
            distances = df_centers - np.expand_dims(superpixel.T, axis=1)
            distances = distances.apply(np.linalg.norm, axis=0)

            # Calculate distance to the closest concept
            distance_to_closest_concept = distances.min()
            closest_concept_to_superpixel = distances.idxmin()
            is_concept_present[closest_concept_to_superpixel] = 1

        # Create image labels
        image_label = [is_concept_present[concept] for concept in top_concepts]
        labels.append(image_label)

    return np.array(labels)


import argparse


def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument("--generate")
    parser.add_argument("--classes", nargs="+", default=[])
    return parser


def main(args=None):
    local_path = ROOT_DIR
    parser = get_parser()
    args = parser.parse_args(args)
    generate = args.generate
    classes = args.classes
    assert generate in [
        "patches",
        "patch_embeddings",
        "activations",
        "labels",
        "embeddings",
    ]
    for clas in classes:
        assert clas in test_classes
    if classes == []:
        classes = test_classes

    if generate == "patches":
        for clas in classes:
            calc_patches("train", clas, "mixed4c")
            calc_patches("val", clas, "mixed4c")
            print(f"Patches for {clas} are saved")

    # elif generate == "patch_embeddings":
    #     for representation in ["byol", "swav", "moco", "simclr"]:
    #         for clas in classes:
    #             for phase in ["train", "val"]:
    #                 create_ss_patch_embeddings(
    #                     os.path.join(local_path, "superpixels/"),
    #                     phase,
    #                     clas,
    #                     representation,
    #                 )

    elif generate == "activations":
        for clas in classes:
            activation_train = calc_image_activations("train", clas, "mixed4c")
            pickle.dump(
                activation_train,
                open(
                    os.path.join(
                        local_path, f"data/activations/train_activations_{clas}.pkl"
                    ),
                    "wb",
                ),
            )

            activation_val = calc_image_activations("val", clas, "mixed4c")
            pickle.dump(
                activation_val,
                open(
                    os.path.join(
                        local_path, f"data/activations/val_activations_{clas}.pkl"
                    ),
                    "wb",
                ),
            )

            print(f"Activations for {clas} are saved")

    elif generate == "labels":
        for clas in classes:
            train_label = create_concept_labels(
                "train",
                clas,
                test_classes,
                top_concepts,
                local_path=os.path.join(local_path, "data"),
            )
            with open(
                os.path.join(local_path, f"data/labels/train_labels_55_{clas}.pkl"),
                "wb",
            ) as labels_file:
                pickle.dump(train_label, labels_file)

            val_label = create_concept_labels(
                "val",
                clas,
                test_classes,
                top_concepts,
                local_path=os.path.join(local_path, "data"),
            )
            with open(
                os.path.join(local_path, f"data/labels/val_labels_55_{clas}.pkl"), "wb"
            ) as labels_file:
                pickle.dump(val_label, labels_file)

            print(f"Labels for {clas} are saved")

    # elif generate == "embeddings":
    #     for representation in ["moco", "byol", "swav", "simclr"]:
    #         create_ss_embeddings(representation, "train", classes)
    #         create_ss_embeddings(representation, "val", classes)


if __name__ == "__main__":
    main()
