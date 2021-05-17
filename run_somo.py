import sys
import os
import numpy as np
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

"""This script uploads and trains probing models for semantic odd man out task."""

def run_somo(repr_type):
    all_embeddings = []
    no_somo_embeddings = []

    val_all_embeddings = []
    val_no_somo_embeddings = []

    classes_55 = os.listdir('./somo/train/')
    for class_name in classes_55:
        train_embeds_temp = np.load(os.path.join(os.path.join('./somo/train', class_name, repr_type+'.npy')))
        all_embeddings.append(train_embeds_temp[650:,:])

        no_somo_train_embeds_temp = np.load(os.path.join('./somo/train', class_name, repr_type+'_no_somo.npy'))
        no_somo_embeddings.append(no_somo_train_embeds_temp[:650,:])
        
        val_swav_embeddings = np.load(os.path.join(os.path.join('./somo/val', class_name, repr_type+'.npy')))
        val_all_embeddings.append(val_swav_embeddings[:25, :])

        val_no_somo_swav_embeddings = np.load(os.path.join('./somo/val', class_name, repr_type+'_no_somo.npy'))
        val_no_somo_embeddings.append(val_no_somo_swav_embeddings[25:, :])
    
    train_negative_embeds = np.concatenate(no_somo_embeddings)
    train_positive_embeds = np.concatenate(all_embeddings)
    
    val_negative_embeds = np.concatenate(val_no_somo_embeddings)
    val_positive_embeds = np.concatenate(val_all_embeddings)

    train_y_negative = np.array([0 for i in range(train_negative_embeds.shape[0])])
    train_y_positive = np.array([1 for i in range(train_positive_embeds.shape[0])])
    
    val_y_negative = np.array([0 for i in range(val_negative_embeds.shape[0])])
    val_y_positive = np.array([1 for i in range(val_positive_embeds.shape[0])])

    x_train = np.concatenate([train_negative_embeds, train_positive_embeds])
    y_train = np.concatenate([train_y_negative, train_y_positive])
    
    x_test = np.concatenate([val_negative_embeds, val_positive_embeds])
    y_test = np.concatenate([val_y_negative, val_y_positive])
    
    # Shuffle
    shuffle_idx = np.random.permutation(x_train.shape[0])
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    
    shuffle_idx = np.random.permutation(x_test.shape[0])
    x_test = x_test[shuffle_idx]
    y_test = y_test[shuffle_idx]
    
        
    log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=80)
    log_reg.fit(x_train, y_train)

    pkl_filename = "probing_results/somo/log_reg_somo_"+repr_type+'.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(log_reg, file)

    y_pred = log_reg.predict(x_test)
    y_pred_proba = log_reg.predict_proba(x_test)

    results = pd.DataFrame.from_dict(classification_report(y_test, y_pred,
                                                            output_dict=True)).round(2)

    results.to_csv('probing_results/somo/'+repr_type+'_somo.csv')

    roc_auc = roc_auc_score(y_test, y_pred_proba[:,1])
    roc_auc_ovo = roc_auc_score(y_test, y_pred_proba[:,1])

    with open('probing_results/somo/'+repr_type+'_somo_roc_auc_ovo.txt', 'w') as file:
        file.write(str(roc_auc_ovo))
    file.close()
    
    with open('probing_results/somo/'+repr_type+'_somo_roc_auc.txt', 'w') as file:
        file.write(str(roc_auc))
    file.close()

    
if __name__ == "__main__":
    print('SIMCLR')
    run_somo('simclr')

    print('BYOL')
    run_somo('byol_2')

    print('MOCO')
    run_somo('moco_2')

    print('SWAV')
    run_somo('swav_2')