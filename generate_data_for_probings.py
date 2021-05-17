import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
IMAGE_NET_PATH = '/local/data/ImageNet'
from PIL import Image
from create_data import _get_in_dirname
import sys
sys.path.append("./ACE")
from ACE.ace_helpers import load_image_from_file
from ACE import ace_helpers
from ACE.ace import ConceptDiscovery
from tcav import utils
from pathlib import Path
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow_hub as hub
from simclr_pytorch import EncodeProject


top_concepts = ['zebra_concept5',
 'dingo_concept8',
 'bison_concept9',
 'koala_concept19',
 'jaguar_concept6',
 'chimpanzee_concept17',
 'hog_concept8',
 'hamster_concept2',
 'lion_concept7',
 'beaver_concept1',
 'lynx_concept3',
 'convertible_concept2',
 'sports_car_concept15',
 'airliner_concept17',
 'jeep_concept1',
 'passenger_car_concept23',
 'steam_locomotive_concept14',
 'cab_concept15',
 'garbage_truck_concept11',
 'warplane_concept10',
 'ambulance_concept13',
 'police_van_concept4',
 'planetarium_concept4',
 'castle_concept14',
 'church_concept5',
 'mosque_concept11',
 'triumphal_arch_concept7',
 'barn_concept9',
 'stupa_concept16',
 'boathouse_concept1',
 'suspension_bridge_concept1',
 'steel_arch_bridge_concept16',
 'viaduct_concept16',
 'sax_concept10',
 'flute_concept3',
 'cornet_concept10',
 'panpipe_concept8',
 'drum_concept11',
 'cello_concept17',
 'acoustic_guitar_concept12',
 'grand_piano_concept11',
 'banjo_concept10',
 'maraca_concept15',
 'chime_concept2',
 'Granny_Smith_concept6',
 'fig_concept16',
 'custard_apple_concept11',
 'banana_concept10',
 'corn_concept6',
 'lemon_concept10',
 'pomegranate_concept1',
 'pineapple_concept12',
 'jackfruit_concept7',
 'strawberry_concept5',
 'orange_concept6',
 'sports_car_concept19',
 'sports_car_concept16',
 'sports_car_concept10',
 'sports_car_concept11',
 'panpipe_concept13',
 'panpipe_concept1',
 'steam_locomotive_concept17',
 'steel_arch_bridge_concept9',
 'orange_concept8',
 'panpipe_concept5',
 'panpipe_concept6',
 'panpipe_concept15',
 'steel_arch_bridge_concept12',
 'orange_concept11',
 'steel_arch_bridge_concept13',
 'panpipe_concept14',
 'zebra_concept7',
 'sports_car_concept2',
 'steel_arch_bridge_concept6',
 'panpipe_concept11',
 'steel_arch_bridge_concept14',
 'sports_car_concept6',
 'steel_arch_bridge_concept1',
 'planetarium_concept2',
 'orange_concept9',
 'panpipe_concept12',
 'panpipe_concept2',
 'steel_arch_bridge_concept11',
 'Granny_Smith_concept8',
 'Granny_Smith_concept10',
 'sports_car_concept9',
 'sports_car_concept7',
 'orange_concept16',
 'triumphal_arch_concept10',
 'stupa_concept11',
 'sports_car_concept18',
 'airliner_concept9',
 'steel_arch_bridge_concept3',
 'mosque_concept12',
 'strawberry_concept3',
 'sports_car_concept12',
 'airliner_concept3',
 'orange_concept15',
 'planetarium_concept14',
 'planetarium_concept18']

classes = ['kit_fox', 'English_setter', 'Siberian_husky', 'Australian_terrier', 'English_springer', 'grey_whale', 'lesser_panda', 'Egyptian_cat', 'ibex', 'Persian_cat', 'cougar', 'gazelle', 'porcupine', 'sea_lion', 'malamute', 'badger', 'Great_Dane', 'Walker_hound', 'Welsh_springer_spaniel', 'whippet', 'Scottish_deerhound', 'killer_whale', 'mink', 'African_elephant', 'Weimaraner', 'soft-coated_wheaten_terrier', 'Dandie_Dinmont', 'red_wolf', 'Old_English_sheepdog', 'jaguar', 'otterhound', 'bloodhound', 'Airedale', 'hyena', 'meerkat', 'giant_schnauzer', 'titi', 'three-toed_sloth', 'sorrel', 'black-footed_ferret', 'dalmatian', 'black-and-tan_coonhound', 'papillon', 'skunk', 'Staffordshire_bullterrier', 'Mexican_hairless', 'Bouvier_des_Flandres', 'weasel', 'miniature_poodle', 'Cardigan', 'malinois', 'bighorn', 'fox_squirrel', 'colobus', 'tiger_cat', 'Lhasa', 'impala', 'coyote', 'Yorkshire_terrier', 'Newfoundland', 'brown_bear', 'red_fox', 'Norwegian_elkhound', 'Rottweiler', 'hartebeest', 'Saluki', 'grey_fox', 'schipperke', 'Pekinese', 'Brabancon_griffon', 'West_Highland_white_terrier', 'Sealyham_terrier', 'guenon', 'mongoose', 'indri', 'tiger', 'Irish_wolfhound', 'wild_boar', 'EntleBucher', 'zebra', 'ram', 'French_bulldog', 'orangutan', 'basenji', 'leopard', 'Bernese_mountain_dog', 'Maltese_dog', 'Norfolk_terrier', 'toy_terrier', 'vizsla', 'cairn', 'squirrel_monkey', 'groenendael', 'clumber', 'Siamese_cat', 'chimpanzee', 'komondor', 'Afghan_hound', 'Japanese_spaniel', 'proboscis_monkey', 'guinea_pig', 'white_wolf', 'ice_bear', 'gorilla', 'borzoi', 'toy_poodle', 'Kerry_blue_terrier', 'ox', 'Scotch_terrier', 'Tibetan_mastiff', 'spider_monkey', 'Doberman', 'Boston_bull', 'Greater_Swiss_Mountain_dog', 'Appenzeller', 'Shih-Tzu', 'Irish_water_spaniel', 'Pomeranian', 'Bedlington_terrier', 'warthog', 'Arabian_camel', 'siamang', 'miniature_schnauzer', 'collie', 'golden_retriever', 'Irish_terrier', 'affenpinscher', 'Border_collie', 'hare', 'boxer', 'silky_terrier', 'beagle', 'Leonberg', 'German_short-haired_pointer', 'patas', 'dhole', 'baboon', 'macaque', 'Chesapeake_Bay_retriever', 'bull_mastiff', 'kuvasz', 'capuchin', 'pug', 'curly-coated_retriever', 'Norwich_terrier', 'flat-coated_retriever', 'hog', 'keeshond', 'Eskimo_dog', 'Brittany_spaniel', 'standard_poodle', 'Lakeland_terrier', 'snow_leopard', 'Gordon_setter', 'dingo', 'standard_schnauzer', 'hamster', 'Tibetan_terrier', 'Arctic_fox', 'wire-haired_fox_terrier', 'basset', 'water_buffalo', 'American_black_bear', 'Angora', 'bison', 'howler_monkey', 'hippopotamus', 'chow', 'giant_panda', 'American_Staffordshire_terrier', 'Shetland_sheepdog', 'Great_Pyrenees', 'Chihuahua', 'tabby', 'marmoset', 'Labrador_retriever', 'Saint_Bernard', 'armadillo', 'Samoyed', 'bluetick', 'redbone', 'polecat', 'marmot', 'kelpie', 'gibbon', 'llama', 'miniature_pinscher', 'wood_rabbit', 'Italian_greyhound', 'lion', 'cocker_spaniel', 'Irish_setter', 'dugong', 'Indian_elephant', 'beaver', 'Sussex_spaniel', 'Pembroke', 'Blenheim_spaniel', 'Madagascar_cat', 'Rhodesian_ridgeback', 'lynx', 'African_hunting_dog', 'langur', 'Ibizan_hound', 'timber_wolf', 'cheetah', 'English_foxhound', 'briard', 'sloth_bear', 'Border_terrier', 'German_shepherd', 'otter', 'koala', 'tusker', 'echidna', 'wallaby', 'platypus', 'wombat', 'revolver', 'umbrella', 'schooner', 'soccer_ball', 'accordion', 'ant', 'starfish', 'chambered_nautilus', 'grand_piano', 'laptop', 'strawberry', 'airliner', 'warplane', 'airship', 'balloon', 'space_shuttle', 'fireboat', 'gondola', 'speedboat', 'lifeboat', 'canoe', 'yawl', 'catamaran', 'trimaran', 'container_ship', 'liner', 'pirate', 'aircraft_carrier', 'submarine', 'wreck', 'half_track', 'tank', 'missile', 'bobsled', 'dogsled', 'bicycle-built-for-two', 'mountain_bike', 'freight_car', 'passenger_car', 'barrow', 'shopping_cart', 'motor_scooter', 'forklift', 'electric_locomotive', 'steam_locomotive', 'amphibian', 'ambulance', 'beach_wagon', 'cab', 'convertible', 'jeep', 'limousine', 'minivan', 'Model_T', 'racer', 'sports_car', 'go-kart', 'golfcart', 'moped', 'snowplow', 'fire_engine', 'garbage_truck', 'pickup', 'tow_truck', 'trailer_truck', 'moving_van', 'police_van', 'recreational_vehicle', 'streetcar', 'snowmobile', 'tractor', 'mobile_home', 'tricycle', 'unicycle', 'horse_cart', 'jinrikisha', 'oxcart', 'bassinet', 'cradle', 'crib', 'four-poster', 'bookcase', 'china_cabinet', 'medicine_chest', 'chiffonier', 'table_lamp', 'file', 'park_bench', 'barber_chair', 'throne', 'folding_chair', 'rocking_chair', 'studio_couch', 'toilet_seat', 'desk', 'pool_table', 'dining_table', 'entertainment_center', 'wardrobe', 'Granny_Smith', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit', 'custard_apple', 'pomegranate', 'acorn', 'hip', 'ear', 'rapeseed', 'corn', 'buckeye', 'organ', 'upright', 'chime', 'drum', 'gong', 'maraca', 'marimba', 'steel_drum', 'banjo', 'cello', 'violin', 'harp', 'acoustic_guitar', 'electric_guitar', 'cornet', 'French_horn', 'trombone', 'harmonica', 'ocarina', 'panpipe', 'bassoon', 'oboe', 'sax', 'flute', 'daisy', "yellow_lady's_slipper", 'cliff', 'valley', 'alp', 'volcano', 'promontory', 'sandbar', 'coral_reef', 'lakeside', 'seashore', 'geyser', 'hatchet', 'cleaver', 'letter_opener', 'plane', 'power_drill', 'lawn_mower', 'hammer', 'corkscrew', 'can_opener', 'plunger', 'screwdriver', 'shovel', 'plow', 'chain_saw', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house_finch', 'junco', 'indigo_bunting', 'robin', 'bulbul', 'jay', 'magpie', 'chickadee', 'water_ouzel', 'kite', 'bald_eagle', 'vulture', 'great_grey_owl', 'black_grouse', 'ptarmigan', 'ruffed_grouse', 'prairie_chicken', 'peacock', 'quail', 'partridge', 'African_grey', 'macaw', 'sulphur-crested_cockatoo', 'lorikeet', 'coucal', 'bee_eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted_merganser', 'goose', 'black_swan', 'white_stork', 'black_stork', 'spoonbill', 'flamingo', 'American_egret', 'little_blue_heron', 'bittern', 'crane', 'limpkin', 'American_coot', 'bustard', 'ruddy_turnstone', 'red-backed_sandpiper', 'redshank', 'dowitcher', 'oystercatcher', 'European_gallinule', 'pelican', 'king_penguin', 'albatross', 'great_white_shark', 'tiger_shark', 'hammerhead', 'electric_ray', 'stingray', 'barracouta', 'coho', 'tench', 'goldfish', 'eel', 'rock_beauty', 'anemone_fish', 'lionfish', 'puffer', 'sturgeon', 'gar', 'loggerhead', 'leatherback_turtle', 'mud_turtle', 'terrapin', 'box_turtle', 'banded_gecko', 'common_iguana', 'American_chameleon', 'whiptail', 'agama', 'frilled_lizard', 'alligator_lizard', 'Gila_monster', 'green_lizard', 'African_chameleon', 'Komodo_dragon', 'triceratops', 'African_crocodile', 'American_alligator', 'thunder_snake', 'ringneck_snake', 'hognose_snake', 'green_snake', 'king_snake', 'garter_snake', 'water_snake', 'vine_snake', 'night_snake', 'boa_constrictor', 'rock_python', 'Indian_cobra', 'green_mamba', 'sea_snake', 'horned_viper', 'diamondback', 'sidewinder', 'European_fire_salamander', 'common_newt', 'eft', 'spotted_salamander', 'axolotl', 'bullfrog', 'tree_frog', 'tailed_frog', 'whistle', 'wing', 'paintbrush', 'hand_blower', 'oxygen_mask', 'snorkel', 'loudspeaker', 'microphone', 'screen', 'mouse', 'electric_fan', 'oil_filter', 'strainer', 'space_heater', 'stove', 'guillotine', 'barometer', 'rule', 'odometer', 'scale', 'analog_clock', 'digital_clock', 'wall_clock', 'hourglass', 'sundial', 'parking_meter', 'stopwatch', 'digital_watch', 'stethoscope', 'syringe', 'magnetic_compass', 'binoculars', 'projector', 'sunglasses', 'loupe', 'radio_telescope', 'bow', 'cannon', 'assault_rifle', 'rifle', 'projectile', 'computer_keyboard', 'typewriter_keyboard', 'crane', 'lighter', 'abacus', 'cash_machine', 'slide_rule', 'desktop_computer', 'hand-held_computer', 'notebook', 'web_site', 'harvester', 'thresher', 'printer', 'slot', 'vending_machine', 'sewing_machine', 'joystick', 'switch', 'hook', 'car_wheel', 'paddlewheel', 'pinwheel', "potter's_wheel", 'gas_pump', 'carousel', 'swing', 'reel', 'radiator', 'puck', 'hard_disc', 'sunglass', 'pick', 'car_mirror', 'solar_dish', 'remote_control', 'disk_brake', 'buckle', 'hair_slide', 'knot', 'combination_lock', 'padlock', 'nail', 'safety_pin', 'screw', 'muzzle', 'seat_belt', 'ski', 'candle', "jack-o'-lantern", 'spotlight', 'torch', 'neck_brace', 'pier', 'tripod', 'maypole', 'mousetrap', 'spider_web', 'trilobite', 'harvestman', 'scorpion', 'black_and_gold_garden_spider', 'barn_spider', 'garden_spider', 'black_widow', 'tarantula', 'wolf_spider', 'tick', 'centipede', 'isopod', 'Dungeness_crab', 'rock_crab', 'fiddler_crab', 'king_crab', 'American_lobster', 'spiny_lobster', 'crayfish', 'hermit_crab', 'tiger_beetle', 'ladybug', 'ground_beetle', 'long-horned_beetle', 'leaf_beetle', 'dung_beetle', 'rhinoceros_beetle', 'weevil', 'fly', 'bee', 'grasshopper', 'cricket', 'walking_stick', 'cockroach', 'mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly', 'damselfly', 'admiral', 'ringlet', 'monarch', 'cabbage_butterfly', 'sulphur_butterfly', 'lycaenid', 'jellyfish', 'sea_anemone', 'brain_coral', 'flatworm', 'nematode', 'conch', 'snail', 'slug', 'sea_slug', 'chiton', 'sea_urchin', 'sea_cucumber', 'iron', 'espresso_maker', 'microwave', 'Dutch_oven', 'rotisserie', 'toaster', 'waffle_iron', 'vacuum', 'dishwasher', 'refrigerator', 'washer', 'Crock_Pot', 'frying_pan', 'wok', 'caldron', 'coffeepot', 'teapot', 'spatula', 'altar', 'triumphal_arch', 'patio', 'steel_arch_bridge', 'suspension_bridge', 'viaduct', 'barn', 'greenhouse', 'palace', 'monastery', 'library', 'apiary', 'boathouse', 'church', 'mosque', 'stupa', 'planetarium', 'restaurant', 'cinema', 'home_theater', 'lumbermill', 'coil', 'obelisk', 'totem_pole', 'castle', 'prison', 'grocery_store', 'bakery', 'barbershop', 'bookshop', 'butcher_shop', 'confectionery', 'shoe_shop', 'tobacco_shop', 'toyshop', 'fountain', 'cliff_dwelling', 'yurt', 'dock', 'brass', 'megalith', 'bannister', 'breakwater', 'dam', 'chainlink_fence', 'picket_fence', 'worm_fence', 'stone_wall', 'grille', 'sliding_door', 'turnstile', 'mountain_tent', 'scoreboard', 'honeycomb', 'plate_rack', 'pedestal', 'beacon', 'mashed_potato', 'bell_pepper', 'head_cabbage', 'broccoli', 'cauliflower', 'zucchini', 'spaghetti_squash', 'acorn_squash', 'butternut_squash', 'cucumber', 'artichoke', 'cardoon', 'mushroom', 'shower_curtain', 'jean', 'carton', 'handkerchief', 'sandal', 'ashcan', 'safe', 'plate', 'necklace', 'croquet_ball', 'fur_coat', 'thimble', 'pajama', 'running_shoe', 'cocktail_shaker', 'chest', 'manhole_cover', 'modem', 'tub', 'tray', 'balance_beam', 'bagel', 'prayer_rug', 'kimono', 'hot_pot', 'whiskey_jug', 'knee_pad', 'book_jacket', 'spindle', 'ski_mask', 'beer_bottle', 'crash_helmet', 'bottlecap', 'tile_roof', 'mask', 'maillot', 'Petri_dish', 'football_helmet', 'bathing_cap', 'teddy', 'holster', 'pop_bottle', 'photocopier', 'vestment', 'crossword_puzzle', 'golf_ball', 'trifle', 'suit', 'water_tower', 'feather_boa', 'cloak', 'red_wine', 'drumstick', 'shield', 'Christmas_stocking', 'hoopskirt', 'menu', 'stage', 'bonnet', 'meat_loaf', 'baseball', 'face_powder', 'scabbard', 'sunscreen', 'beer_glass', 'hen-of-the-woods', 'guacamole', 'lampshade', 'wool', 'hay', 'bow_tie', 'mailbag', 'water_jug', 'bucket', 'dishrag', 'soup_bowl', 'eggnog', 'mortar', 'trench_coat', 'paddle', 'chain', 'swab', 'mixing_bowl', 'potpie', 'wine_bottle', 'shoji', 'bulletproof_vest', 'drilling_platform', 'binder', 'cardigan', 'sweatshirt', 'pot', 'birdhouse', 'hamper', 'ping-pong_ball', 'pencil_box', 'pay-phone', 'consomme', 'apron', 'punching_bag', 'backpack', 'groom', 'bearskin', 'pencil_sharpener', 'broom', 'mosquito_net', 'abaya', 'mortarboard', 'poncho', 'crutch', 'Polaroid_camera', 'space_bar', 'cup', 'racket', 'traffic_light', 'quill', 'radio', 'dough', 'cuirass', 'military_uniform', 'lipstick', 'shower_cap', 'monitor', 'oscilloscope', 'mitten', 'brassiere', 'French_loaf', 'vase', 'milk_can', 'rugby_ball', 'paper_towel', 'earthstar', 'envelope', 'miniskirt', 'cowboy_hat', 'trolleybus', 'perfume', 'bathtub', 'hotdog', 'coral_fungus', 'bullet_train', 'pillow', 'toilet_tissue', 'cassette', "carpenter's_kit", 'ladle', 'stinkhorn', 'lotion', 'hair_spray', 'academic_gown', 'dome', 'crate', 'wig', 'burrito', 'pill_bottle', 'chain_mail', 'theater_curtain', 'window_shade', 'barrel', 'washbasin', 'ballpoint', 'basketball', 'bath_towel', 'cowboy_boot', 'gown', 'window_screen', 'agaric', 'cellular_telephone', 'nipple', 'barbell', 'mailbox', 'lab_coat', 'fire_screen', 'minibus', 'packet', 'maze', 'pole', 'horizontal_bar', 'sombrero', 'pickelhaube', 'rain_barrel', 'wallet', 'cassette_player', 'comic_book', 'piggy_bank', 'street_sign', 'bell_cote', 'fountain_pen', 'Windsor_tie', 'volleyball', 'overskirt', 'sarong', 'purse', 'bolo_tie', 'bib', 'parachute', 'sleeping_bag', 'television', 'swimming_trunks', 'measuring_cup', 'espresso', 'pizza', 'breastplate', 'shopping_basket', 'wooden_spoon', 'saltshaker', 'chocolate_sauce', 'ballplayer', 'goblet', 'gyromitra', 'stretcher', 'water_bottle', 'dial_telephone', 'soap_dispenser', 'jersey', 'school_bus', 'jigsaw_puzzle', 'plastic_bag', 'reflex_camera', 'diaper', 'Band_Aid', 'ice_lolly', 'velvet', 'tennis_ball', 'gasmask', 'doormat', 'Loafer', 'ice_cream', 'pretzel', 'quilt', 'maillot', 'tape_player', 'clog', 'iPod', 'bolete', 'scuba_diver', 'pitcher', 'matchstick', 'bikini', 'sock', 'CD_player', 'lens_cap', 'thatch', 'vault', 'beaker', 'bubble', 'cheeseburger', 'parallel_bars', 'flagpole', 'coffee_mug', 'rubber_eraser', 'stole', 'carbonara', 'dumbbell']
assert classes.__len__() == 1000

test_classes = ["zebra", "dingo", "bison", "koala", "jaguar", "chimpanzee", "hog", "hamster", "lion", "beaver", "lynx", "convertible", "sports_car", "airliner", "jeep", "passenger_car", "steam_locomotive", "cab", "garbage_truck", "warplane", "ambulance", "police_van", "planetarium", "castle", "church", "mosque", "triumphal_arch", "barn", "stupa", "boathouse", "suspension_bridge", "steel_arch_bridge", "viaduct", "sax", "flute", "cornet", "panpipe", "drum", "cello", "acoustic_guitar", "grand_piano", "banjo", "maraca", "chime", "Granny_Smith", "fig", "custard_apple", "banana", "corn", "lemon", "pomegranate", "pineapple", "jackfruit", "strawberry", "orange"]

assert test_classes.__len__() == 55

class ImagenetDataset(Dataset):

    def __init__(self, root_path: str, phase: str, target_class: str, transforms=None):
        self.root_path = root_path
        self.phase = phase
        self.target_class = target_class
        self.transforms = transforms
        self.target_dirname = _get_in_dirname(self.target_class)
        self.directory = os.path.join(self.root_path, self.phase, self.target_dirname)
        filenames = [filename for filename in filter(lambda x: x.endswith(".JPEG"), os.listdir(self.directory))]
        self.filenames = [os.path.join(self.directory, filename) for filename in filenames]

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
                    img = img[:,:,:3]

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
                    self.filenames.append(os.path.join(self.path, target_class, file_name))

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
                    img = img[:,:,:3]
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
        for file_name in sorted(os.listdir(os.path.join(self.path, phase, target_class))):
            if file_name.endswith(".png"):
                self.filenames.append(os.path.join(self.path, phase, target_class, file_name))

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
                    img = img[:,:,:3]
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

def calc_image_activations(phase:str,
                           target_class:str,
                           bottlenecks,
                           image_net_path='/local/data/ImageNet',
                           model_to_run="GoogleNet",
                           model_path="ACE/tensorflow_inception_graph.pb",
                           labels_path="ACE/imagenet_labels.txt"):
    
    # Make tensorflow model
    sess = utils.create_session()
    model = ace_helpers.make_model(sess, model_to_run, model_path, labels_path)
    
    # Create dummy ConceptDiscovery object
    # We will need it to generate image activations
    dummy_cd = ConceptDiscovery(model=model,
                                target_class=None,
                                random_concept=None,
                                bottlenecks=bottlenecks,
                                sess=None,
                                source_dir=None,
                                activation_dir=None,
                                cav_dir=None,
                                num_workers=16)
    
    # Create Dataset object
    ds = ImagenetDataset(image_net_path,
                         phase,
                         target_class)
    
    # Save activations
    activations = []
    
    # Iterate through images
    for idx, image in enumerate(ds):
        
        # Extract superpixels embeddings
        sp_outputs = dummy_cd._return_superpixels(image, 'slic', param_dict ={'n_segments': [15, 50, 80]})
        image_superpixels, image_patches = sp_outputs
        superpixels_embedding = ace_helpers.get_acts_from_images(image_superpixels, model, dummy_cd.bottlenecks[0])
        superpixels_embedding = np.mean(superpixels_embedding, axis=(1, 2))
        activations.append(superpixels_embedding)

    return activations

def calc_patches(phase:str,
                 target_class:str,
                 bottlenecks,
                 image_net_path='/local/data/ImageNet',
                 model_to_run="GoogleNet",
                 model_path="ACE/tensorflow_inception_graph.pb",
                 labels_path="ACE/imagenet_labels.txt",
                 path="/local/data/oleszkie"):
    
    # Make tensorflow model
    sess = utils.create_session()
    model = ace_helpers.make_model(sess, model_to_run, model_path, labels_path)
    
    # Create dummy ConceptDiscovery object
    # We will need it to generate image activations
    dummy_cd = ConceptDiscovery(model=model,
                                target_class=None,
                                random_concept=None,
                                bottlenecks=bottlenecks,
                                sess=None,
                                source_dir=None,
                                activation_dir=None,
                                cav_dir=None,
                                num_workers=16)
    
    # Create Dataset object
    ds = ImagenetDataset(image_net_path,
                         phase,
                         target_class)

    # Iterate through images
    for idx, image in enumerate(ds):

        # Extract superpixels embeddings
        sp_outputs = dummy_cd._return_superpixels(image, 'slic', param_dict ={'n_segments': [15, 50, 80]})
        image_superpixels, image_patches = sp_outputs
        for sp_idx, superpixel in enumerate(image_superpixels):
            superpixel = (np.clip(superpixel, 0, 1) * 256).astype(np.uint8)
            with open(os.path.join(path, f"/superpixels/{phase}_superpixels/{target_class}/{idx:06}_{sp_idx:06}.png", "wb")) as sp_file:
                Image.fromarray(superpixel).save(sp_file, format='PNG')
        for patch_idx, patch in enumerate(image_patches):
            patch = (np.clip(patch, 0, 1) * 256).astype(np.uint8)
            with open(os.path.join(path, f"/patches/{phase}_patches/{target_class}/{idx:06}_{patch_idx:06}.png", "wb")) as patch_file:
                Image.fromarray(patch).save(patch_file, format='PNG')
    return


def create_ss_patch_embeddings(path, phase, target_class, representation, local_path="/local/data/oleszkie"):
    
    # Load self-supervised model
    model = None
    sess = None
    
    if representation == "swav":
        model = torch.hub.load('facebookresearch/swav', 'resnet50')
        # We don't need output layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
    
    elif representation == "simclr":
        model = hub.Module('gs://simclr-checkpoints/simclrv2/pretrained/r50_1x_sk0/hub/', trainable=False)
    
    elif representation == "byol":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ResNet('resnet50', pretrained=False, use_fc=False).to(device)
        
        # load encoder
        model_path = os.path.join(local_path, 'models/resnet50_byol_imagenet2012.pth.tar')
        checkpoint = torch.load(model_path, map_location=device)['online_backbone']
        state_dict = {}
        length = len(model.encoder.state_dict())
        for name, param in zip(model.encoder.state_dict(), list(checkpoint.values())[:length]):
            state_dict[name] = param
        model.encoder.load_state_dict(state_dict, strict=True)
        model.eval()
        
    elif representation == "moco":
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.__dict__['resnet50']().to(device)

        # load pretrained weights from the file
        model_path = os.path.join(local_path, 'models/moco_v1_200ep_pretrain.pth.tar')
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['state_dict']

        # preprocess state dict
        for key in list(state_dict.keys()):
            if key.startswith('module.encoder_q') and not key.startswith('module.encoder_q.fc'):
                state_dict[key[len('module.encoder_q.'):]] = state_dict[key]
            del state_dict[key]

        # load model
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        
        # We don't need output layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        
        model.eval()

    model = model.cuda()
    
    if representation in ["swav", "byol", "moco"] :
        eval_model = lambda img: model(img).detach().cpu().numpy().squeeze()


    # Calulcate self-supervised embeddings
    
    ds = PatchesDataset(path, phase, target_class, transforms.Compose([transforms.ToTensor()]))
    #ds = SomoDataset(path, phase, target_class, transforms.Compose([transforms.ToTensor()]))
    loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=20)
    
    result = None
    for idx, image in enumerate(loader):
        image = image.cuda()
        if result is None:
            result = eval_model(image)
        else:
            batch_result = eval_model(image)
            result = np.vstack((result, batch_result))
        
    if sess:
        sess.close()

    # Serialzie
    save_path = os.path.join(path, f"{phase}_superpixels", target_class, f"{representation}_{cut_layers}")
    np.save(save_path, result)
    print(f"Results {result.shape} saved at {save_path}")


def create_ss_embeddings(representation: str, phase: str, target_classes, local_path="/local/data/oleszkie"):
    
    # Load self-supervised model

    model = None
    sess = None
    
    if representation == "swav":
        model = torch.hub.load('facebookresearch/swav', 'resnet50')
        # We don't need output layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
    
    elif representation == "simclr":
        model = hub.Module('gs://simclr-checkpoints/simclrv2/pretrained/r50_1x_sk0/hub/', trainable=False)
        
    
    elif representation == "byol":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ResNet('resnet50', pretrained=False, use_fc=False).to(device)
        
        # load encoder
        model_path = os.path.join(local_path, 'models/resnet50_byol_imagenet2012.pth.tar')
        checkpoint = torch.load(model_path, map_location=device)['online_backbone']
        state_dict = {}
        length = len(model.encoder.state_dict())
        for name, param in zip(model.encoder.state_dict(), list(checkpoint.values())[:length]):
            state_dict[name] = param
        model.encoder.load_state_dict(state_dict, strict=True)
        model.eval()
        
    elif representation == "moco":
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.__dict__['resnet50']().to(device)

        # load pretrained weights from the file
        model_path = os.path.join(local_path, '/models/moco_v1_200ep_pretrain.pth.tar')
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['state_dict']

        # preprocess state dict
        for key in list(state_dict.keys()):
            if key.startswith('module.encoder_q') and not key.startswith('module.encoder_q.fc'):
                state_dict[key[len('module.encoder_q.'):]] = state_dict[key]
            del state_dict[key]

        # load model
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        
        # We don't need output layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        
        model.eval()

    model = model.cuda()
    
    for target_class in target_classes:

        if representation in ["swav", "byol", "moco"] :
            eval_model = lambda img: model(img).detach().cpu().numpy().squeeze()

        elif representation == "simclr":
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            eval_model = lambda img: model(np.moveaxis(img.cpu().numpy(), 1, 3)).eval(session=sess)

        # Create dataset object
        ds = ImagenetDataset(IMAGE_NET_PATH,
                             phase,
                             target_class,
                             transforms.Compose([transforms.ToTensor()]))
        loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=16)

        # Calulcate self-supervised embeddings
        result = None
        for idx, image in enumerate(loader):
            image = image.cuda()
            print(target_class+" "+str(idx))
            if result is None:
                result = eval_model(image)
            else:
                result = np.vstack((result, eval_model(image)))
            print(result.shape)

        # Serialzie
        with open(os.path.join(local_path, 'embeddings/{phase}_embd_{representation}_55_{target_class}.pkl'), "wb") as file:
            pickle.dump(result, file)
        if sess:
            sess.close()
        print(f"Saved {result.shape} at {local_path}/embeddings/{phase}_embd_{representation}_55_{target_class}.pkl")
    return


def create_concept_labels(phase: str,
                          target_class: str,
                          all_classes: List[str],
                          top_concepts: List[str],
                          local_path = "/local/data/oleszkie"
                          ):
    
    # Generate filenames with concept centers
    centers_pkls=[os.path.join(local_path, f"/concepts/{clas}_dict.pkl") for clas in all_classes]
    
    # Get dict of (concept_name, concept_center)
    centers = {}

    # Read file with concept centers and radiuses
    for centers_pkl in centers_pkls:
        with open(centers_pkl, "rb") as centers_file:
            centers_pkl = pickle.load(centers_file)
    
            # Parse file with concept centers and radiuses
            for concept_name in centers_pkl['concepts']:
                if concept_name in top_concepts:
                    center = centers_pkl[concept_name+"_"+"center"]
                    centers[concept_name] = center

    
    # Convert dict to pandas dataframe
    df_centers = pd.DataFrame.from_dict(centers)
    
    # List of final labels
    labels = []
    
    # Iterate through activations
    activations = None
    with open(os.path.join(local_path, f"/activations/{phase}_activations_{target_class}.pkl"), "rb") as activations_file:
        activations = pickle.load(activations_file)

    for idx, image in enumerate(activations):
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
    parser = argparse.ArgumentParser('Parser')
    parser.add_argument('--generate')
    parser.add_argument("--classes", nargs="+", default=[])
    return parser


def main(args=None):
    local_path = "/local/data/oleszkie/"
    parser = get_parser()
    args = parser.parse_args(args)
    generate = args.generate
    classes = args.classes
    assert generate in ['patches', "patch_embeddings", 'activations', 'labels', 'embeddings']
    for clas in classes:
        assert clas in test_classes
    if classes == []:
        classes = test_classes
    
    
    if generate == 'patches':
        for clas in classes:
            calc_patches("train", clas, "mixed4c")
            calc_patches("val", clas, "mixed4c")
            print(f"Patches for {clas} are saved")
            
    elif generate == "patch_embeddings":
        for representation in ["byol", "swav", "moco", "simclr"]:
            for clas in classes:
                for phase in ['train', 'val']:
                    create_ss_patch_embeddings(os.path.join(local_path, "superpixels/"), phase, clas, representation)
                        
    
    elif generate == 'activations':
        for clas in classes:
            activation_train = calc_image_activations("train", clas, "mixed4c")
            pickle.dump(activation_train, open(os.path.join(local_path, f"activations/train_activations_{clas}.pkl"), "wb"))
            
            activation_val = calc_image_activations("val", clas, "mixed4c")
            pickle.dump(activation_val, open(os.path.join(local_path, f"activations/val_activations_{clas}.pkl"), "wb"))
            
            print(f"Activations for {clas} are saved")

    elif generate == 'labels':
        for clas in classes:
            train_label = create_concept_labels("train", clas, test_classes, top_concepts)
            with open(os.path.join(local_path, f"labels/train_labels_55_{clas}.pkl"), "wb") as labels_file:
                pickle.dump(train_label, labels_file)
            
            val_label = create_concept_labels("val", clas, test_classes, top_concepts)
            with open(os.path.join(local_path, f"labels/val_labels_55_{clas}.pkl"), "wb") as labels_file:
                pickle.dump(val_label, labels_file)
            
            print(f"Labels for {clas} are saved")
    
    elif generate == 'embeddings':
        for representation in ["moco", "byol", "swav", "simclr"]:    
            create_ss_embeddings(representation, "train", classes)
            create_ss_embeddings(representation, "val", classes)


main()
