import sys
import os
import pickle
from tqdm import tqdm
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

"""Script for preparation of representations of superpixels.
Provide path to folder with superpixels and its embeddings """

def calculate_concept_pixels(image):
    cnt_all_pixels = 0 
    concept_pixels = 0

    for pixel in image.getdata():
        cnt_all_pixels += 1
        if pixel != (117, 117, 117):
            concept_pixels += 1
    return cnt_all_pixels, concept_pixels

    
def calculate_superpixels(path_to_superpixels, train=True):
    if train:
        typ='train'
    else:
        typ='val'
    
    all_embeddings = []
    all_cnt_all_pixels = []
    all_concept_pixels = []

    classes_55 = ["zebra", "dingo", "bison", "koala", "jaguar", "chimpanzee", "hog", "hamster", "lion", "beaver", "lynx", "convertible", "sports_car", "airliner", "jeep", "passenger_car", "steam_locomotive", "cab", "garbage_truck", "warplane", "ambulance", "police_van", "planetarium", "castle", "church", "mosque", "triumphal_arch", "barn", "stupa", "boathouse", "suspension_bridge", "steel_arch_bridge", "viaduct", "sax", "flute", "cornet", "panpipe", "drum", "cello", "acoustic_guitar", "grand_piano", "banjo", "maraca", "chime", "Granny_Smith", "fig", "custard_apple", "banana", "corn", "lemon", "pomegranate", "pineapple", "jackfruit", "strawberry", "orange"]


    for class_name in tqdm(classes_55):
        embedding_file = os.path.join(path_to_superpixels, class_name, 'byol.npy')
        embeddings = np.load(embedding_file)
        all_embeddings.append(embeddings)
        class_list = os.listdir(path_to_superpixels+class_name)
        embedded_files = sorted([file for file in class_list if file.endswith('.png')])[:embeddings.shape[0]]

        for superpixel_file in tqdm(embedded_files):
            superpixel = Image.open(os.path.join(path_to_superpixels, class_name, superpixel_file))
            # Calculate number of concept pixels
            cnt_all_pixels, concept_pixels = calculate_concept_pixels(superpixel)
            all_cnt_all_pixels.append(cnt_all_pixels)
            all_concept_pixels.append(concept_pixels)


    with open('{}_superpixels_byol_all_embeddings.pkl'.format(typ), 'wb') as file:
        pickle.dump(all_embeddings, file)

    with open('{}_superpixels_byol_all_cnt_all_pixels.pkl'.format(typ), 'wb') as file:
        pickle.dump(all_cnt_all_pixels, file)

    with open('{}_superpixels_byol_all_concept_pixels.pkl'.format(typ), 'wb') as file:
        pickle.dump(all_concept_pixels, file)
        
        
if __name__ == "__main__":
    calculate_superpixels('./train_superpixels', train=True)
    calculate_superpixels('./val_superpixels', train=False)