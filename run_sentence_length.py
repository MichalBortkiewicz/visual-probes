import os
import pickle
import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

"""This script uploads embeddings and trains probing models for sentence length task."""

def load_all_representations(simclr_representations, embed_prefix, label_prefix, repr_type):

    all_images = np.zeros((1, 1024))
    all_labels = np.zeros((1, 100))
    test_classes = ["zebra", "dingo", "bison", "koala", "jaguar", "chimpanzee", "hog", "hamster", "lion", "beaver", "lynx", "convertible", "sports_car", "airliner", "jeep", "passenger_car", "steam_locomotive", "cab", "garbage_truck", "warplane", "ambulance", "police_van", "planetarium", "castle", "church", "mosque", "triumphal_arch", "barn", "stupa", "boathouse", "suspension_bridge", "steel_arch_bridge", "viaduct", "sax", "flute", "cornet", "panpipe", "drum", "cello", "acoustic_guitar", "grand_piano", "banjo", "maraca", "chime", "Granny_Smith", "fig", "custard_apple", "banana", "corn", "lemon", "pomegranate", "pineapple", "jackfruit", "strawberry", "orange"]

    for class_name in test_classes:
        representation_file = embed_prefix+class_name+'.pkl'
#         class_name = re.sub(embed_prefix, '', representation_file).split('.')[:-1][0]
        # Load array with representations
        with open(os.path.join('./embeddings', representation_file), 'rb') as img_file:
            tmp_images_file = pickle.load(img_file)
        # Load corresponding array with labels
        with open(os.path.join(os.path.join('./labels', label_prefix+class_name+'.pkl')), 'rb') as label_file:
            tmp_labels_file = pickle.load(label_file)

        all_images = np.concatenate([all_images, tmp_images_file])
        all_labels = np.concatenate([all_labels, tmp_labels_file])
        
    return all_images[1:], all_labels[1:]

def run_sent_len(repr_type, train_prefix, val_prefix):
    # Read representation
    train_representations = [file for file in os.listdir('./embeddings') if train_prefix in file]
    val_representations = [file for file in os.listdir('./labels') if val_prefix in file]
    x_train, y_train = load_all_representations(train_representations, train_prefix+'_', 'train_labels_55_', repr_type)
    x_valid, y_valid = load_all_representations(val_representations, val_prefix+'_', 'val_labels_55_', repr_type)
    
    y_train = y_train.sum(axis=1)
    y_valid = y_valid.sum(axis=1)
    
    bin_mapping = np.array([ 3. , 12.6, 22.2, 31.8, 41.4])
    
    y_train_binned = [y_ex-1 for y_ex in np.digitize(y_train, bin_mapping)]
    y_valid_binned = [y_ex-1 for y_ex in np.digitize(y_valid, bin_mapping)]
    
    log_reg = LogisticRegression(class_weight='balanced', max_iter=5000, multi_class='ovr', n_jobs=80)
    log_reg.fit(x_train, y_train_binned)
    
    pkl_filename = "probing_results/log_reg_sentence_length_"+repr_type+'.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(log_reg, file)
    
    y_pred = log_reg.predict(x_valid)
    y_pred_proba = log_reg.predict_proba(x_valid)
    
    results = pd.DataFrame.from_dict(classification_report(y_valid_binned, y_pred,
                                                        output_dict=True)).round(2)
    
    results.to_csv('probing_results/'+repr_type+'_sent_length.csv')
    
    roc_auc = roc_auc_score(y_valid_binned, y_pred_proba, multi_class='ovr')
    roc_auc_ovo = roc_auc_score(y_valid_binned, y_pred_proba, multi_class='ovo')
    
    with open('probing_results/'+repr_type+'_roc_auc_ovo.txt', 'w') as file:
        file.write(str(roc_auc_ovo))
    file.close()
    
    with open('probing_results/'+repr_type+'_roc_auc.txt', 'w') as file:
        file.write(str(roc_auc))
    file.close()
    

if __name__ == "__main__":
    print('SIMCLR')
    run_sent_len('simclr', 'train_embd_simclr_55', 'val_embd_simclr_55')
    print('MOCO')
    run_sent_len('moco', 'train_embd_moco_55', 'val_embd_moco_55')
    print('BYOL')
    run_sent_len('byol', 'train_embd_byol_55', 'val_embd_byol_55')
    print('SWAV')
    run_sent_len('swav', 'train_embd_swav_55', 'val_embd_swav_55')