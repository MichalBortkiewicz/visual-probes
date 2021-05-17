import sys
import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import re
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split




def run_character_bin(repr_type, train_prefix, val_prefix):
    with open(train_prefix+'_all_embeddings.pkl', 'rb') as file:
        all_embeddings = pickle.load(file)

    with open(train_prefix+'_all_concept_pixels.pkl', 'rb') as file:
        all_concept_pixels = pickle.load(file)
        
    with open(val_prefix+'_all_embeddings.pkl', 'rb') as file:
        val_all_embeddings = pickle.load(file)

    with open(val_prefix+'_all_concept_pixels.pkl', 'rb') as file:
        val_all_concept_pixels = pickle.load(file)
        
    all_embeddings_swav = np.concatenate(all_embeddings)
    val_all_embeddings_swav = np.concatenate(val_all_embeddings)
    
    bin_mapping = np.array([ 2194., 10191., 18188., 26185., 34182., 42179., 50176.])
    
    y_train = [y_ex-1 for y_ex in np.digitize(all_concept_pixels, bin_mapping)]
    y_test = [y_ex-1 for y_ex in np.digitize(val_all_concept_pixels, bin_mapping)]
    
    x_train = all_embeddings_swav
    x_test = val_all_embeddings_swav
    
    log_reg = LogisticRegression(class_weight='balanced', max_iter=500, multi_class='ovr', n_jobs=80)
    log_reg.fit(x_train, y_train)
    
    pkl_filename = "probing_results/log_reg_char_bin_"+repr_type+'.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(log_reg, file)
    
    y_pred = log_reg.predict(x_test)
    y_pred_proba = log_reg.predict_proba(x_test)
    
    results = pd.DataFrame.from_dict(classification_report(y_test, y_pred,
                                                        output_dict=True)).round(2)
    
    results.to_csv('probing_results/'+repr_type+'_char_bin.csv')
    
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    roc_auc_ovo = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')
    with open('probing_results/'+repr_type+'_char_bin_roc_auc.txt', 'w') as file:
        file.write(str(roc_auc))
    file.close()
    with open('probing_results/'+repr_type+'_char_bin_roc_auc_ovo.txt', 'w') as file:
        file.write(str(roc_auc_ovo))
    file.close()

if __name__ == "__main__":
    print('SWAV')
    run_character_bin('swav', 'train_swav', 'val_swav')

    print('MOCO')
    run_character_bin('moco', 'train_superpixels_moco', 'val_superpixels_moco')

    print('BYOL')
    run_character_bin('byol', 'train_superpixels_byol', 'val_superpixels_byol')

    print('SIMCLR')
    run_character_bin('simclr', 'train_superpixels_simclr', 'val_superpixels_simclr')
