# Everything is copied over from molecular/train_isensee2017.py on 04/13/2021 and modified according to rcnn requirements
import copy
import os
import glob
import pathlib
import pprint
import random
import re
import shutil
import string
import time
import pandas as pd
import pickle

from sklearn.model_selection import StratifiedKFold


random.seed(9001)
import numpy as np

import nibabel as nib
from pathlib import Path
import importlib
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import logging
import multiprocessing as mp

from collections import Counter
import random

import skimage.transform
from nilearn.image import new_img_like
from nilearn.image import reorder_img, new_img_like
import SimpleITK as sitk
from sklearn.utils import class_weight
from sklearn.preprocessing import OneHotEncoder

import visualize



def return_none():
    return None

def get_class_ids_from_df(df, config, sessions):
    if os.path.isabs(sessions[0]):
        sessions_basename = [os.path.basename(sess) for sess in sessions]
    else:
        sessions_basename = sessions

    y = [df.loc[df.my_id == session_basename][config['marker_column']].item() for session_basename in sessions_basename]

    # for i,j in zip(get_basename_from_abspath(sessions), y):
    #     print(i,"-->", j)

    return dict(Counter(y)), y

def get_basename_from_abspath(list_of_abspaths):
    if os.path.isabs(list_of_abspaths[0]):
        list_of_basenames = [os.path.basename(sess) for sess in list_of_abspaths]
        return list_of_basenames
    else:
        return list_of_abspaths

def set_data_paths(config, fold, ext_exp, exp):

    basepath = "/scratch/satrajit.chakrabarty/molecular_experiments/Exp" + exp + "/" + "fold" + fold + "/"

    # Path to which training/validation/test hdf5 files will be written to
    config["data_file_tr"] = os.path.abspath(basepath + "fold{}_data_tr.h5".format(fold))
    config["data_file_val"] = os.path.abspath(basepath + "fold{}_data_val.h5".format(fold))
    
    # Path to which pickle files containing training/validation/test indices will be written to
    config["training_file"] = os.path.abspath(basepath + "training_ids.pkl")
    config["validation_file"] = os.path.abspath(basepath + "validation_ids.pkl")

   
    config["data_file_test"] = os.path.abspath(basepath + "fold{}_data_test.h5".format(fold))
    config["testing_file"] = os.path.abspath(basepath + "testing_ids.pkl")


    # Path to which pickle files containing external test indices will be written to
    # Path to which external test hdf5 files will be written to
    config["data_file_ext"] = "/scratch/satrajit.chakrabarty/molecular_experiments/Exp" + ext_exp + "/" + "data_ext.h5"
    config["ext_file"] = "/scratch/satrajit.chakrabarty/molecular_experiments/Exp" + ext_exp + "/" + "ext_ids.pkl"


def plot_crosstabs(df, fold, marker_column, savepath):
    crosstab_vars = ["Histology", "Grade", "cohort", marker_column ]
    fig, axes = plt.subplots(1,len(crosstab_vars), figsize = (10*len(crosstab_vars),10))

    for ax, variable in zip(axes.ravel(), crosstab_vars):
        sns.heatmap(pd.crosstab(df["scratch_path"], df[variable], margins=True), ax = ax, square = True, annot_kws={"size": 30}, cmap = 'YlOrBr', annot=True, cbar=False, fmt="d",
            linewidths=1, linecolor='black')

        ax.yaxis.label.set_visible(False)
        ax.set_title(variable, fontsize = '25')

        # We change the fontsize of minor ticks label 
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig(savepath + 'fold{}_info_crosstabs.png'.format(fold))
    plt.close()

def plot_info(info, fold, savepath):

    # barplot : # sessions before vs after modality omission
    if 'sessions_before_modality_omission' in info.keys() and 'sessions_after_modality_omission' in info.keys():
        ax = plt.subplot()
        xlabels = ['before', 'after']
        ylabels = [len(info['sessions_before_modality_omission']), len(info['sessions_after_modality_omission'])]

        ax.bar(xlabels,ylabels)

        rects = ax.patches

        for rect, label in zip(rects, ylabels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
                    ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(savepath + 'fold{}_info_bef_after_mod_omit.png'.format(fold))
        plt.close()

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    if 'sessions_per_fold' in info.keys():
        labels = ["fold" + str(i) for i in range(1,6)]
        sizes = [len(fold) for fold in info['sessions_per_fold']]

        fig1, ax1 = plt.subplots(figsize=(10, 10))
        ax1.pie(sizes, labels=labels, shadow=True, autopct=lambda pct: func(pct, sizes), startangle=90, textprops=dict(size=40))
        # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.tight_layout()
        plt.savefig(savepath + 'info_split_per_fold.png')
        plt.close()

    # Stacked bar plot
    class_distribution_keys = [i for i in list(info.keys()) if 'class_distribution' in i]

    D_list = []
    class_keys = list(eval(info[class_distribution_keys[0]]).keys())

    # print(class_distribution_keys)

    for class_distribution_key in class_distribution_keys:
        D = eval(info[class_distribution_key])
        D_list.append(list(D.values()))

    C = np.array(D_list).T.tolist()

    # Source: https://stackoverflow.com/questions/41296313/stacked-bar-chart-with-centered-labels
    df = pd.DataFrame(dict(zip(class_keys, C)))

    ax = df.plot(stacked=True, kind='bar', figsize=(12, 8), rot='horizontal')

    # .patches is everything inside of the chart
    for rect in ax.patches:
        # Find where everything is located
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()

        # The height of the bar is the data value and can be used as the label
        label_text = int(height)  # f'{height:.2f}' to format decimal values

        # ax.text(x, y, text)
        label_x = x + width / 2
        label_y = y + height / 2

        # plot only when height is greater than specified value
        if height > 0:
            ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=20, color='w')

    ax.legend(loc='best', fontsize=30)
    ax.set_ylabel("Count", fontsize=20)
    ax.set_xlabel("Class", fontsize=20)

    xticklabelslist = [i.split('_')[-1] for i in class_distribution_keys]

    ax.set_xticklabels(xticklabelslist, fontsize=15)

    plt.tight_layout()
    plt.savefig(savepath + 'fold{}_info_class_dist_per_fold.png'.format(fold))
    plt.close()


def sequence_dropout(files, subject_ids, config, train0val1):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sequence dropout ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    drop_fraction = config["seq_dropout_fraction_tr"] if train0val1 == 0 else config["seq_dropout_fraction_val"]
    number_of_sessions = len(files) # Let's say this is 50
    number_of_sessions_to_drop_sequence = int(number_of_sessions*drop_fraction) # this is 25

    list_of_session_ids = list(range(number_of_sessions)) # [0, 1, 2, ...49]
    list_of_session_ids_to_drop_sequence = random.sample(list_of_session_ids, number_of_sessions_to_drop_sequence) # randomly chosen 25 elements from list_of_session_ids
    list_of_session_ids_to_drop_ch1 = random.sample(list_of_session_ids_to_drop_sequence, number_of_sessions_to_drop_sequence//2) # from list_of_session_ids_to_drop_sequence, choose half of it to drop channel 1
    list_of_session_ids_to_drop_ch2 = list(set(list_of_session_ids_to_drop_sequence).difference(list_of_session_ids_to_drop_ch1)) # ... and then choose the rest to drop channel 2

    return (list_of_session_ids_to_drop_ch1, list_of_session_ids_to_drop_ch2)

def create_training_validation_testing_files(config, df, path_to_sessions, print_fn = print):
    training_files = list()
    subject_ids_tr = list()

    for subject_dir in path_to_sessions:
        subject_ids_tr.append(os.path.basename(subject_dir))
        subject_files = list()
        # if 'LGG1p19q' not in subject_dir:
        for modality in config["training_modalities"] + config["truth"]: subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        # else:
        #     for modality in config["training_modalities"] + ["OTMultiClass_partial"]: subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))

        training_files.append(tuple(subject_files))

    training_files = [list(i) for i in training_files] # converting to list of lists from list of tuples

    print_fn("[SUBJECT_IDS] " + str(len(subject_ids_tr)) + " " + str(subject_ids_tr))

    session_labels = np.array([df[df['my_id'] == i][config['marker_column']].iloc[0] for i in subject_ids_tr])
    assert len(session_labels) == len(subject_ids_tr)

    return training_files, session_labels, subject_ids_tr

def trim_df_by_dropping_nans(df, config):
    # Prepare the dataframe by reading from excel file

    # Drop all columns except following
    df.drop(df.columns.difference(['subj',
                                'scratch_path',
                                'my_id',
                                'Histology',
                                'seg_groundtruth',
                                'Grade',
                                'Survival (months)',
                                'Age (years at diagnosis)',
                                config['marker_column']]),
                                1, inplace=True)


    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
    df = df[df.subj.notna()]  # There are some unnecessary rows (like count of data). Remove by using the condition: 'subj'=NaN
    df = df[df.my_id.notna()]  # There are some cases for which we do not have imaging data (i.e. NA in my_id column) - drop them
    df = df[df.scratch_path.notna()]  # There are some cases for which we do not have imaging data (i.e. NA in my_id column) - drop them
    df = df[df['Age (years at diagnosis)'].notna()]  # There are some cases for which we do not have age info
    df = df[df[config['marker_column']].notna()]  # There are some cases without molecular status (i.e. NA in column) - drop them
    df = df[df[config['marker_column']].isin(config['labels_to_use'])] # For marker column, keep only data which has class label in config['labels_to_use']
    df = df[~df.my_id.str.contains("pending")]  # There are some cases that are pending (specified in my_id column). Remove those

    # There are some cases that are pending in molecular status (specified by 'todo' in config['marker_column'] column). Remove those
    # Get names of indexes for which column value is 'todo' and delete these rows
    indexNames = df[df[config['marker_column']] == 'todo'].index
    df.drop(indexNames, inplace=True)

    return df

def trim_df_based_on_GT(df, config, exclude_cases_with_partial_GT = True):

    if "OTMultiClass" in config["training_modalities"]:
        if exclude_cases_with_partial_GT:
            # keep a check that if "OTMulticlass" is included then only take cases with GT, filter from excel
            df = df[df['seg_groundtruth'] == 'Yes']
        else:
            df = df[df['seg_groundtruth'].isin(['Yes', 'partial'])]

    if "bbox_crop_or_mask" not in config:
        config["bbox_crop_or_mask"] = None

    if config["bbox_crop_or_mask"] is not None:
        df = df[df['seg_groundtruth'] == 'Yes']

    return df

def trim_df_based_on_Tumor_modality(df, config):

    # keep a check that if "Tumor_modality" is included then only take cases with GT & T1c, filter from excel
    if "Tumor_modality" in config["training_modalities"]:
        df = df[(df['seg_groundtruth'] == 'Yes') & (df['T1c'] == 'Yes')]

    return df

def check_or_create_tumor_modality(sessions):
    print("[DEBUG] inside check_or_create_tumor_modality")
    # print(*sessions, sep="\n")
    for session in sessions:
        print(session)
        sess_list_of_modalities = glob.glob(os.path.join(session, "*.nii.gz"))
        # print("sess_list_of_modalities",sess_list_of_modalities)
        is_tumor_modality_boolean_list = ["Tumor_modality" in s for s in sess_list_of_modalities]

        if any(is_tumor_modality_boolean_list):
            print("Tumor modality already exists")
        else:
            print("[DISCLAIMER] Tumor_modality did not exist. Creating it by masking OTMulticlass on T1c_subtrMeanDivStd.nii.gz of this subject: ")

            brainmask = np.rint(nib.load(os.path.join(session, 'brainmask.nii.gz')).get_fdata())
            print("{brainmask}", check_unique_elements(brainmask))

            t1c = nib.load(os.path.join(session, 'T1c_subtrMeanDivStd.nii.gz')).get_fdata()
            print("{t1c}", np.count_nonzero(t1c))

            GT = nib.load(os.path.join(session, 'OTMultiClass.nii.gz')).get_fdata().astype('int32')
            print("{GT}", check_unique_elements(GT))
            print("{GT}", np.count_nonzero(GT))

            GT[GT > 0] = 1
            pseudo = GT * t1c
            # pseudo = pseudo * brainmask
            print("{pseudo}", pseudo[0,0,0])

            save_numpy_2_nifti(pseudo,os.path.join(session, 'T1c_subtrMeanDivStd.nii.gz'),os.path.join(session, 'Tumor_modality.nii.gz'))


def trim_df_based_on_presence_in_scratch_and_modality(df, config):

    all_modalities = copy.deepcopy(config["training_modalities"])
    if "Tumor_modality" in all_modalities: all_modalities.remove("Tumor_modality")

    # List of all sessions of the worksheet : ['abspath/to/session1', 'abspath/to/session2', ..., 'abspath/to/sessionn']
    sessions_abspath_all = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['my_id'])) for index, row in df.iterrows()]

    # True/False if session folder exists/doesnt in datapath : [True, False, ..., True]
    session_exists_logical = [os.path.isdir(i) for i in sessions_abspath_all]

    # Subset of sessions_abspath_all, containing only those sessions that exist
    sessions_abspath_exists_bef_modality_check = np.array(sessions_abspath_all)[np.array(session_exists_logical)].tolist()


    session_exists_modality_exists_logical_sublist = [[os.path.exists(os.path.join(i, j + ".nii.gz")) for j in all_modalities] for i in sessions_abspath_exists_bef_modality_check]

    # For each session, this gives True if all req modalities exist for that session
    session_exists_modality_exists_logical = [all(i) for i in session_exists_modality_exists_logical_sublist]

    # Use session_exists_modality_exists_logical indices to filter sessions_abspath_exists_bef_modality_check
    # This is the final list of sessions to be used
    sessions_abspath_exists = np.array(sessions_abspath_exists_bef_modality_check)[np.array(session_exists_modality_exists_logical)].tolist()

    df = df[df['my_id'].isin(os.path.basename(i) for i in (sessions_abspath_exists))]

    return df


def split_data_into_n_folds(config, info, df, sessions_abspath_exists, n_fold=5):

    sessions_abspath_exists_basename = [os.path.basename(i) for i in sessions_abspath_exists]
    y = df.loc[df.my_id.isin(sessions_abspath_exists_basename)][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions

    y_unique, y_count = np.unique(y, return_counts=True)
    info['class_distribution_overall'] = str(dict(zip(y_unique, y_count)))

    # Stratified k-fold sampling
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)  # define sampler

    train_index_list_5folds = []
    val_test_index_list_5folds = []

    for train_index, val_test_index in skf.split(sessions_abspath_exists, y):
        train_index_list_5folds.append(train_index)
        val_test_index_list_5folds.append(val_test_index)

    foldx_sessions = [np.array(sessions_abspath_exists)[idx].tolist() for idx in val_test_index_list_5folds]

    info['sessions_per_fold'] = [[os.path.basename(i) for i in j] for j in foldx_sessions]

    for i, foldi in zip(["fold" + str(i) for i in range(1, n_fold + 1)], foldx_sessions):
        sessname = [os.path.basename(sess) for sess in foldi]
        y_fold = df.loc[df.my_id.isin(sessname)][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions
        y_fold_unique, y_fold_count = np.unique(y_fold, return_counts=True)
        info['class_distribution_' + i] = str(dict(zip(y_fold_unique, y_fold_count)))

    return foldx_sessions


def split_data_into_n_folds_from_excel(config, df):

    training_df = df[df['cohort'] == 'train']
    val_df = df[df['cohort'] == 'val']
    test_df = df[df['cohort'] == 'test']
    ext_df = df[df['cohort'] == 'ext']

    training_session_abs_path = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['my_id'])) for index, row in training_df.iterrows()]
    val_session_abs_path = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['my_id'])) for index, row in val_df.iterrows()]
    test_session_abs_path = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['my_id'])) for index, row in test_df.iterrows()]
    ext_session_abs_path = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['my_id'])) for index, row in ext_df.iterrows()]

    return training_session_abs_path, val_session_abs_path, test_session_abs_path, ext_session_abs_path

def split_data_into_n_folds_from_path(config, info, data_dir):

    val_fold = "fold" + config['fold'] + "/"
    train_fold = "fold[!" + config['fold'] + "]/"

    data_dir_abspath = os.path.join(config["path_to_data"], data_dir)

    training_session_abs_path = glob.glob(os.path.join(data_dir_abspath, train_fold, "*"))
    val_session_abs_path = glob.glob(os.path.join(data_dir_abspath, val_fold, "*"))
    test_session_abs_path = glob.glob(os.path.join(data_dir_abspath, val_fold, "*"))

    return training_session_abs_path, val_session_abs_path, test_session_abs_path

def split_data_into_n_folds_on_subject_level(config, info, df, sessions_abspath_exists, n_fold=5):

    sessions_abspath_exists_basename = [os.path.basename(i) for i in sessions_abspath_exists]
    y = df.loc[df.my_id.isin(sessions_abspath_exists_basename)][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions


    y_unique, y_count = np.unique(y, return_counts=True)
    info['class_distribution_overall'] = str(dict(zip(y_unique, y_count)))

    # sessions_abspath_exists = [subj1_session1, subj1_session2, subj2_session1, subj3_session1, ...]
    # subjs = [subj1, subj1, subj2, subj3, ...]
    # dict_subj_labels = {subj1: class1, subj2: class0, subj3: class0, ...}
    subjs = [i.split('/')[-1].split('_')[0] for i in sessions_abspath_exists]

    dict_subj_labels = dict(zip(subjs,y))

    subjs_unique = list(dict_subj_labels.keys())
    y_unique = list(dict_subj_labels.values())

    sessions_grouped_by_subj = [[i for i in sessions_abspath_exists if i.split('/')[-1].split('_')[0] == x] for x in subjs_unique]
    dict_sessions_grouped_by_subj = dict(zip(subjs_unique, sessions_grouped_by_subj))

    for idx, i in enumerate(subjs):
        class_from_dict = dict_subj_labels[str(i)]
        class_from_list = y[idx]
        assert class_from_dict == class_from_list, "Error! Subject {} has conflicting class-labels from different sessions".format(i)

    # Stratified k-fold sampling
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)  # define sampler

    foldx_index_list = []

    for _, foldx_index in skf.split(subjs_unique, y_unique):
        foldx_index_list.append(foldx_index)

    foldx_subjs = [np.array(subjs_unique)[idx].tolist() for idx in foldx_index_list]

    foldx_sessions = []
    for foldn_subjs in foldx_subjs:
        foldn_sessions = []
        for subj in foldn_subjs:
            foldn_sessions.append(dict_sessions_grouped_by_subj[subj])
        foldn_sessions = [j for i in foldn_sessions for j in i]
        foldx_sessions.append(foldn_sessions)


    info['sessions_per_fold'] = [[os.path.basename(i) for i in j] for j in foldx_sessions]

    for i, foldi in zip(["fold" + str(i) for i in range(1, n_fold + 1)], foldx_sessions):
        sessname = [os.path.basename(sess) for sess in foldi]
        y_fold = df.loc[df.my_id.isin(sessname)][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions
        y_fold_unique, y_fold_count = np.unique(y_fold, return_counts=True)
        info['class_distribution_' + i] = str(dict(zip(y_fold_unique, y_fold_count)))

    return foldx_sessions

def split_val_test(config, info, df, foldx_sessions, fold):
    config['validation_testing_sessions'] = foldx_sessions.pop(int(fold) - 1)
    info['validation_testing_sessions'] = [os.path.basename(sess) for sess in config['validation_testing_sessions']]

    y_val_test = df.loc[df.my_id.isin(info['validation_testing_sessions'])][config['marker_column']].tolist()

    val_index_list_5folds = []
    test_index_list_5folds = []

    # sampler for splitting the validation data into validation and testing
    skf_val_test = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    for val_index, test_index in skf_val_test.split(config['validation_testing_sessions'], y_val_test):
        val_index_list_5folds.append(val_index)
        test_index_list_5folds.append(test_index)

    foldx_sessions_val_test = [np.array(config['validation_testing_sessions'])[idx].tolist() for idx in test_index_list_5folds]

    return foldx_sessions_val_test

def split_val_test_on_subject_level(config, info, df, foldx_sessions, fold):
    config['validation_testing_sessions'] = foldx_sessions.pop(int(fold) - 1)
    info['validation_testing_sessions'] = [os.path.basename(sess) for sess in config['validation_testing_sessions']]

    y_val_test = df.loc[df.my_id.isin(info['validation_testing_sessions'])][config['marker_column']].tolist()

    # sessions_abspath_exists = [subj1_session1, subj1_session2, subj2_session1, subj3_session1, ...]
    # subjs = [subj1, subj1, subj2, subj3, ...]
    # dict_subj_labels = {subj1: class1, subj2: class0, subj3: class0, ...}
    subjs = [i.split('/')[-1].split('_')[0] for i in config['validation_testing_sessions']]

    dict_subj_labels = dict(zip(subjs,y_val_test))

    subjs_unique = list(dict_subj_labels.keys())
    y_unique = list(dict_subj_labels.values())

    sessions_grouped_by_subj = [[i for i in config['validation_testing_sessions'] if i.split('/')[-1].split('_')[0] == x] for x in subjs_unique]
    dict_sessions_grouped_by_subj = dict(zip(subjs_unique, sessions_grouped_by_subj))

    for idx, i in enumerate(subjs):
        class_from_dict = dict_subj_labels[str(i)]
        class_from_list = y_val_test[idx]
        assert class_from_dict == class_from_list, "Error! Subject {} has conflicting class-labels from different sessions".format(i)

    test_index_list_5folds = []

    # sampler for splitting the validation data into validation and testing
    skf_val_test = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    for _, test_index in skf_val_test.split(subjs_unique, y_unique):
        test_index_list_5folds.append(test_index)

    foldx_subjs_val_test = [np.array(subjs_unique)[idx].tolist() for idx in test_index_list_5folds]

    foldx_sessions_val_test = []

    for foldn_subjs in foldx_subjs_val_test:
        foldn_sessions = []
        for subj in foldn_subjs:
            foldn_sessions.append(dict_sessions_grouped_by_subj[subj])
        foldn_sessions = [j for i in foldn_sessions for j in i]
        foldx_sessions_val_test.append(foldn_sessions)


    return foldx_sessions_val_test

def match_all_classes_to_least_represented_class(df, config, tr_sess, testing_sessions):

    # print(*get_basename_from_abspath(tr_sess), sep="\n")

    tr_class_count, tr_classes = get_class_ids_from_df(df, config, tr_sess)
    # print("tr_class_count ", tr_class_count, tr_classes)

    tr_classes_unique = list(set(tr_classes))
    tr_least_represented_class = min(tr_class_count, key=tr_class_count.get)
    # print("tr_least_represented_class: ", tr_least_represented_class)

    tr_sess_with_classes = [(i,j) for i,j in zip(tr_sess, tr_classes)]
    tr_sess_grouped = [[i[0] for i in tr_sess_with_classes if i[1] == class_id] for class_id in tr_classes_unique]
    # print("tr_sess_grouped", tr_sess_grouped)
    # print("tr_sess_grouped[0]", len(tr_sess_grouped[0]))
    # print(*get_basename_from_abspath(tr_sess_grouped[0]), sep="\n")
    # print("tr_sess_grouped[1]", len(tr_sess_grouped[1]))
    # print(*get_basename_from_abspath(tr_sess_grouped[1]), sep="\n")
    tr_sess_grouped_dict = dict(zip(tr_classes_unique, tr_sess_grouped))

    tr_n_sess_with_least_represented_class = len(tr_sess_grouped_dict[tr_least_represented_class])
    # print("tr_n_sess_with_least_represented_class: ", tr_n_sess_with_least_represented_class)

    spilled_cases = []

    for key,value in tr_sess_grouped_dict.items():
        # print("key {}: # cases = {}, dist = {}".format(key,len(value), get_class_ids_from_df(df, config, value)[0]))

        if key is not tr_least_represented_class:
            random.shuffle(value)

            # for i,j in zip(get_basename_from_abspath(value), get_class_ids_from_df(df, config, value)[1]):
                # print(i,"-->", j)

            keep_sess, spill_sess = value[:tr_n_sess_with_least_represented_class], value[tr_n_sess_with_least_represented_class:]
            # print("keep_sess", get_class_ids_from_df(df, config, keep_sess)[0])
            # print("spill_sess", get_class_ids_from_df(df, config, spill_sess)[0])
            tr_sess_grouped_dict[key] = keep_sess
            spilled_cases.extend(spill_sess)

    tr_sess_grouped_matched = list(tr_sess_grouped_dict.values())
    training_sessions_matched = [j for i in tr_sess_grouped_matched for j in i]
    testing_sessions.extend(spilled_cases)

    # print(get_class_ids_from_df(df, config, training_sessions_matched)[0])

    # dsad
    return training_sessions_matched, testing_sessions


def get_suptitle(df,session_name, fold, config):

    session_name = re.sub('_pos[0-9]', '', session_name)
    session_name = re.sub('_neg[0-9]', '', session_name)
    session_name = re.sub('_mid', '', session_name)

    my_dict = df.loc[df['my_id'] == session_name].iloc[0].to_dict()
    my_dict['fold'] = fold

    fig_sup_title = []
    # for key, value in my_dict.items():
    #     if key in ["my_id", "seg_groundtruth", "Histology", "Grade", 'Survival (months)', "cohort", "fold", config["marker_column"]]:
    #         fig_sup_title = fig_sup_title + [str(key) + ' : ' + str(value)]

    for key in ["my_id", "seg_groundtruth", "Histology", "Grade", 'Age (years at diagnosis)', 'Survival (months)', "cohort", "fold", config['marker_column']]:
        if key in my_dict.keys():
            fig_sup_title = fig_sup_title + [str(key) + ' : ' + str(my_dict[key])]

    # print("fig_sup_title", fig_sup_title)

    fig_sup_title = [fig_sup_title[:2]] + [fig_sup_title[2:-1]] + [fig_sup_title[-1]]
    fig_sup_title = [str(i) for i in fig_sup_title]
    return fig_sup_title

def get_suptitle_with_verdict(df, session_name, verdict, confidence, fold, config):
    my_dict = df.loc[df['my_id'] == session_name].iloc[0].to_dict()
    my_dict['fold'] = fold
    # print(my_dict)
    fig_sup_title = []
    for key in ["my_id", "seg_groundtruth", "Histology", "Grade", 'Age (years at diagnosis)', 'Survival (months)', "cohort", "fold", config['marker_column']]:
        if key in my_dict.keys():
            fig_sup_title = fig_sup_title + [str(key) + ' : ' + str(my_dict[key])]

    fig_sup_title = [fig_sup_title[:2]] + [fig_sup_title[2:-1]] + [fig_sup_title[-1]]
    fig_sup_title = [str(i) for i in fig_sup_title]
    return fig_sup_title

def print_data_info(logger, cf, config, info):
    logger.info("~" * 60 + " [CONFIG] " + "~" * 60)

    for line in pprint.pformat(cf.__dict__).split('\n'):
        logger.debug(line)

    logger.info("~" * 60 + " [DATA DISTRIBUTION] " + "~" * 60)

    try:
        logger.info("\n" +"[INFO] Total #sessions before exclusion based on modality: " + str(len(info['sessions_before_modality_omission'])))
    except: pass

    try:
        logger.info("[INFO] Total #sessions after exclusion based on modality: " + str(len(info['sessions_after_modality_omission'])))
    except: pass

    try:
        logger.info("[INFO] Total #sessions (internal cohort): " + str(len(config['training_sessions'])+len(config['validation_sessions'])+len(config['testing_sessions'])) + " " + str(info['class_distribution_int']))
    except: pass

    try:
        logger.info("[INFO] Total #sessions (training): "+ str(len(config['training_sessions'])) + " " + str(info['class_distribution_training']))
    except: pass


    try:
        logger.info("[INFO] Total #sessions (validation): "+ str(len(config['validation_sessions'])) + " "+ str(info['class_distribution_validation']))
    except: pass


    try:
        logger.info("[INFO] Total #sessions (testing): "+ str(len(config['testing_sessions'])) + " "+ str(info['class_distribution_testing']))
    except: pass

    try:
        logger.info("[INFO] Total #sessions (external cohort): "+ str(len(config['ext_sessions'])) + " "+str(info['class_distribution_ext']))
    except: pass


def create_logger(config):
    log_path = os.path.join(config["basepath"], "training_log.txt")
    LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(message)s"
    logging.basicConfig(filename=log_path,
                        filemode='w',
                        format=LOG_FORMAT,
                        level=logging.DEBUG)

    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path)
    logger.addHandler(fh)

    logging.getLogger('matplotlib.font_manager').disabled = True

    return logger


def create_basepath_and_code_snapshot(fold,config,config_file_name,info,df):
    # Create the basepath folder if it does not already exist
    if not os.path.exists(config["basepath"]):
        pathlib.Path(config["basepath"]).mkdir(parents=True, exist_ok=True)

    df.to_csv(os.path.join(config["basepath"], 'df_filtered.csv'), index=False)

    with open(os.path.join(config["basepath"], 'fold{}_info.pickle'.format(fold)), 'wb') as handle:
        pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the config as a pickle file
    with open(os.path.join(config["basepath"], 'fold{}_config.pickle'.format(fold)), 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # snapshot current code
    if not os.path.exists(os.path.join(config["basepath"],'code_snapshot/')):
        pathlib.Path(os.path.join(config["basepath"],'code_snapshot/')).mkdir(parents=True, exist_ok=True)

    shutil.copy2(__file__, os.path.abspath(os.path.join(config["basepath"],'code_snapshot')))
    shutil.copy2(os.path.join(Path(os.path.dirname(__file__)), 'config_files', config_file_name + ".py"), os.path.abspath(os.path.join(config["basepath"],'code_snapshot')))


# Transformation for mapping
def transform_from_axial(vol, target_plane):
    """
    Function to transform volume into Axial axis and back
    :param np.ndarray vol: image volume to transform
    :param bool coronal2axial: transform from coronal to axial = True (default),
                               transform from axial to coronal = False
    :return:
    """
    if target_plane == 'coronal':
        return np.moveaxis(vol, [0, 1, 2], [1, 2, 0])
    elif target_plane == 'sagittal':
        return np.moveaxis(vol, [0, 1, 2], [2, 0, 1])
    elif target_plane == 'axial':
        return vol
    else: raise ValueError(f'Plane variable "{target_plane}" not recognized')


########## Resize 2d slice of nifti  ##########

def resize_wrapper(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """
    copied from matterport maskrcnn
    """
    # This if for skimage version > 0.14: anti_aliasing. Default it to False for backward
    # compatibility with skimage 0.13.
    return skimage.transform.resize(
        image, output_shape,
        order=order, mode=mode, cval=cval, clip=clip,
        preserve_range=preserve_range, anti_aliasing=anti_aliasing,
        anti_aliasing_sigma=anti_aliasing_sigma)


def resize_nifti_slice(image, min_dim, max_dim, mode="square"):
    """
    copied from matterport maskrcnn
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))


    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize_wrapper(image, (round(h * scale), round(w * scale)), preserve_range=True)

    # Return square image
    
    # Get new height and width
    h, w = image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)

    return image.astype(image_dtype)

def get_test_csv_from_result_csv(cf, cohort_suffix, model_suffix, exp_op_dir):
    path_to_cohort_data = cf.cohort_dict[cohort_suffix]['path_to_data']
    p_df = pd.read_pickle(os.path.join(path_to_cohort_data, cf.input_df_name))

    p_df['pid'] = pd.to_numeric(p_df['pid'])
    p_df['class_id'] = p_df['class_id'] + 1

    # for each patient keep only the prediction with highest score
    results_csv = pd.read_csv(os.path.join(exp_op_dir, f'results_{model_suffix}+{cohort_suffix}.csv'))
    results_csv = results_csv.sort_values('score').drop_duplicates('patientID', keep='last')
    results_csv = results_csv.rename(columns={'patientID': 'pid'})


    results_gt_df = pd.merge(p_df, results_csv, on='pid')
    results_gt_df.drop(columns=['path', 'predictionID'], inplace = True)

    results_gt_df['Verdict'] = np.where(results_gt_df['class_id']==results_gt_df['pred_classID'], 'correct', 'wrong')
    results_gt_df = results_gt_df.rename(columns={'subj_id': 'my_id'})

    df_merged = pd.merge(cf.df_filtered, results_gt_df, on='my_id')
    df_merged['class_id'].replace(cf.class_dict, inplace = True)
    df_merged['pred_classID'].replace(cf.class_dict, inplace = True)
    df_merged = df_merged.rename(columns={'pred_classID': 'Prediction'})

    df_merged.to_csv(os.path.join(exp_op_dir, f'prediction_scores_{model_suffix}+{cohort_suffix}.csv'), index = False)

########## Resize 3d volume of nifti  ##########
def calculate_origin_offset(new_spacing, old_spacing):
    return np.subtract(new_spacing, old_spacing)/2


def sitk_resample_to_spacing(image, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear, default_value=0.):
    zoom_factor = np.divide(image.GetSpacing(), new_spacing)
    new_size = np.asarray(np.ceil(np.round(np.multiply(zoom_factor, image.GetSize()), decimals=5)), dtype=np.int16)
    offset = calculate_origin_offset(new_spacing, image.GetSpacing())
    reference_image = sitk_new_blank_image(size=new_size, spacing=new_spacing, direction=image.GetDirection(),
                                           origin=image.GetOrigin() + offset, default_value=default_value)
    return sitk_resample_to_image(image, reference_image, interpolator=interpolator, default_value=default_value)


def sitk_resample_to_image(image, reference_image, default_value=0., interpolator=sitk.sitkLinear, transform=None,
                           output_pixel_type=None):
    if transform is None:
        transform = sitk.Transform()
        transform.SetIdentity()
    if output_pixel_type is None:
        output_pixel_type = image.GetPixelID()
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetTransform(transform)
    resample_filter.SetOutputPixelType(output_pixel_type)
    resample_filter.SetDefaultPixelValue(default_value)
    resample_filter.SetReferenceImage(reference_image)
    return resample_filter.Execute(image)


def sitk_new_blank_image(size, spacing, direction, origin, default_value=0.):
    image = sitk.GetImageFromArray(np.ones(size, dtype=np.float).T * default_value)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image


def resample_to_spacing(data, spacing, target_spacing, interpolation="linear", default_value=0.):
    image = data_to_sitk_image(data, spacing=spacing)
    if interpolation is "linear":
        interpolator = sitk.sitkLinear
    elif interpolation is "nearest":
        interpolator = sitk.sitkNearestNeighbor
    else:
        raise ValueError("'interpolation' must be either 'linear' or 'nearest'. '{}' is not recognized".format(
            interpolation))
    resampled_image = sitk_resample_to_spacing(image, new_spacing=target_spacing, interpolator=interpolator,
                                               default_value=default_value)
    return sitk_image_to_data(resampled_image)


def data_to_sitk_image(data, spacing=(1., 1., 1.)):
    if len(data.shape) == 3:
        data = np.rot90(data, 1, axes=(0, 2))
    image = sitk.GetImageFromArray(data)
    image.SetSpacing(np.asarray(spacing, dtype=np.float))
    return image


def sitk_image_to_data(image):
    data = sitk.GetArrayFromImage(image)
    if len(data.shape) == 3:
        data = np.rot90(data, -1, axes=(0, 2))
    return data



def resize_nifti_vol(image, new_shape, interpolation):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine)

# [Dummy code for volume resize]
# T1c = nib.load('./T1c_subtrMeanDivStd.nii.gz')
# T1c_resized = resize(T1c, (128,128,128), interpolation = "linear" )
# T1c_resized.to_filename('./T1c_resized.nii.gz')

# T1c = nib.load('./OTMultiClass.nii.gz')
# T1c_resized = resize(T1c, (128,128,128), interpolation = "nearest" )
# T1c_resized.to_filename('./OTMultiClass_resized.nii.gz')

def get_npy_from_df_for_roc(cf, df):

    pred_class = df['Prediction'].tolist()
    truth = df[cf.config['marker_column']].tolist()
    confidence = df['score'].tolist()

    predicted_class_idx = [cf.config['labels_to_use'].index(pred) for pred in pred_class]

    pred_zeros = [[0,0] for i in range(len(predicted_class_idx))]

    for pix, pred_i in enumerate(pred_zeros):
        pred_i[predicted_class_idx[pix]] = confidence[pix]
        pred_i[pred_i.index(0)] = 1 - confidence[pix]

    y_pred = np.array(pred_zeros)    

    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(np.array(truth).reshape(-1,1))
    y_true = np.array(onehot_encoded)

    return y_true, y_pred

def plot_ROC_PR_CM_from_df(cf, exp_op_dir, cohort_suffix, model_suffix):

    df = pd.read_csv(os.path.join(exp_op_dir, f'prediction_scores_{model_suffix}+{cohort_suffix}.csv'))

    fig, axes = plt.subplots(1, 3, figsize=(24,8))
    ax_ROC, ax_PR, ax_CM = axes
    y_true, y_pred = get_npy_from_df_for_roc(cf, df)
    visualize.plot_roc(cf.config, y_true, y_pred , cohort_suffix, axis = ax_ROC)
    visualize.plot_precision_recall(cf.config, y_true, y_pred , cohort_suffix, axis = ax_PR)
    visualize.plot_cm_from_df(df, cf.config, ax = ax_CM)

    # Save plot
    plt.savefig(os.path.join(exp_op_dir, f'ROC_PR_CM_{model_suffix}+{cohort_suffix}.png'), bbox_inches = 'tight')
    plt.close()


def check_unique_elements(np_array):
    # Extract the end-points of the 3D bbox from the tumor mask
    unique, counts = np.unique(np_array, return_counts = True)
    return dict(zip(unique,counts))

    
def compute_class_weights(df_filtered, config):

    subject_ids = get_basename_from_abspath(config['training_sessions'])
    y_train = df_filtered.loc[df_filtered['my_id'].isin(subject_ids)][config['marker_column']].tolist()
    class_weights = class_weight.compute_class_weight('balanced', classes = config['labels_to_use'], y=y_train)
    final_class_weights = [1] + (class_weights/class_weights.min()).tolist() # [1, 1.46, 1] for 'BG', Mutant', 'WT' respectively

    print("\n" + "[CLASS_WEIGHTS] " + str(dict(zip(['BG'] + config['labels_to_use'], final_class_weights))))

    return final_class_weights


def fix_canonical(image):
    file_ort = nib.aff2axcodes(image.affine)
    
    if file_ort != ('R','A','S'):
        print("Converting to canonical (RAS orientation)")
        return nib.as_closest_canonical(image)
    else:
        # print("Image already canonical (RAS orientation)")
        return image