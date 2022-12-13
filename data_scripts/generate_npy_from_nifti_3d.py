#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import numpy as np
import pandas as pd
import pickle
from multiprocessing import Pool
import nibabel as nib
import sys
import multiprocessing
import argparse
import importlib
from utils.config_utils import *


################################################################
# healthy = 0, necrotic = 1, edema = 2, enhancing = 4
################################################################

def multi_processing_create_image(inputs):

    out_dir, path_to_data, class_label, subj_id, idx, cf = inputs

    try:
        mask_path = glob.glob(path_to_data[-1])[0]
    except:
        mask_path = glob.glob(os.path.join('/scratch/satrajit.chakrabarty/segmentations_128x128x128/', 'ext_'+subj_id, "prediction_upsampled.nii.gz"))[0]
        
    seg_vol = nib.load(mask_path)
    assert nib.aff2axcodes(seg_vol.affine) == ('R','A','S'), "Mask is not in RAS space."

    seg_vol = resize_nifti_vol(seg_vol, (128,128,128), interpolation = "nearest").get_fdata()
    seg_vol = np.rot90(seg_vol)
    seg = seg_vol
    
    seg[seg > 0] = 1

    image_array = []

    for i in path_to_data[:-1]:
        img_vol = nib.load(i)
        img_vol = fix_canonical(img_vol)
        img_vol = resize_nifti_vol(img_vol, (128,128,128), interpolation = "linear").get_fdata()
        img_vol = np.rot90(img_vol)
        img = img_vol
        image_array.append(img)

    img = np.stack(image_array, axis=0)

    if seg.shape[0] == 128: seg = np.expand_dims(seg, axis = 0)
    if img.shape[0] == 128: img = np.expand_dims(img, axis = 0)
    out = np.concatenate((img, seg), axis = 0)
    print('\nprocessing pid={}, subj_id = {}, \nimg = {}, \nseg = {}, \nout = {}'.format(idx, subj_id, path_to_data[0], mask_path, out.shape))
    out_path = os.path.join(out_dir, '{}.npy'.format(idx))
    np.save(out_path, out)

    class_id = cf.class_assignment[class_label]
    
    with open(os.path.join(out_dir, 'meta_info_{}.pickle'.format(idx)), 'wb') as handle:
        pickle.dump([subj_id, out_path, class_id, str(idx)], handle)


def generate_experiment(cf, cohort_files, cohort_labels, subject_ids_cohort, cohort_suffix = 'train', print_fn = print):

    cohort_dir = cf.cohort_dict[cohort_suffix]['path_to_data']
    
    if not os.path.exists(cohort_dir): os.makedirs(cohort_dir)

    info = []
    n_train_images = len(subject_ids_cohort)

    info += [[cohort_dir, path_to_data, class_label, subj_id, idx, cf] for path_to_data, class_label, subj_id, idx in zip(cohort_files, cohort_labels, subject_ids_cohort, range(n_train_images))]

    n_workers = 2 if os.name is 'nt' else int(multiprocessing.cpu_count()//2)
    print_fn(f"Using {n_workers} workers to generate data")
    print_fn('Saving data at = {}'.format(cohort_dir))

    pool = Pool(processes=n_workers)
    pool.map(multi_processing_create_image, info, chunksize=1)
    pool.close()
    pool.join()

    aggregate_meta_info(cohort_dir)


def aggregate_meta_info(exp_dir):

    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['subj_id', 'path', 'class_id', 'pid'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
    print ("aggregated meta info to df with length", len(df))

    # clean up meta_info files
    for f in files: os.remove(f)




def prepare_data_for_cohort(cf, path_to_sessions, cohort_suffix, debugmode=None, print_fn = print):
    print_fn(f"\n {'=' * 30} [{cohort_suffix} SUBJECT IDS] {'=' * 30}")
    cohort_files, cohort_labels, subject_ids_cohort = create_training_validation_testing_files(cf.config,
                                                                                               cf.df_filtered,
                                                                                               path_to_sessions=path_to_sessions,
                                                                                               print_fn = print_fn)

    generate_experiment(cf, cohort_files, cohort_labels, subject_ids_cohort, cohort_suffix = cohort_suffix, print_fn = print_fn)





def main(args, cohort_suffix = 'train', print_fn=print):

    exp_folder = f"{str(args.mol_stat)}"
    cfg_name = f"{args.exp}"

    config_file = importlib.import_module(f"config_files.{exp_folder}."+ cfg_name)

    cf = config_file.configs(args.dim)

    if cohort_suffix == 'train':
        path_to_sessions = cf.config["training_sessions"]
    elif cohort_suffix == 'test':
        path_to_sessions = cf.config["testing_sessions"]
    elif cohort_suffix == 'WUSM':
        path_to_sessions = cf.config["ext_sessions"]
    elif cohort_suffix == 'EGD':
        path_to_sessions = cf.config["EGD_sessions"]
    else:
        raise ValueError(f"cohort_suffix = {cohort_suffix} not recognized")

    prepare_data_for_cohort(cf,
                            path_to_sessions = path_to_sessions,
                            cohort_suffix = cohort_suffix,
                            print_fn=print_fn)

