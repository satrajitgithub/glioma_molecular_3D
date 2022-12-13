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

"""Default Configurations script. Avoids changing configs of all experiments if general settings are to be changed."""

import os
import multiprocessing as mp 
from collections import defaultdict
from utils.config_utils import *

class DefaultConfigs:

    def __init__(self, model, dim, molecular_status, server_env=None):

        #########################
        #    Session paths      #
        #########################

        self.config = defaultdict(return_none)
        self.info = dict()

        # Set the molecular parameter for this experiment
        molecular_zoo = {'IDH': {'labels_to_use': ['Mutant', 'WT'],  # mutant, Wild-type
                                'worksheet_to_use': 'molecular_public',
                                'marker_column': 'IDH_status'}
                        }

        self.config['molecular_status'] = molecular_status

        try:
            molecular_params_dict = molecular_zoo[self.config['molecular_status']]
        except KeyError:
            raise ValueError(f"Could not find details for {molecular_status} in molecular_zoo. Instead choose from: {list(molecular_zoo.keys())}")

        for key in molecular_params_dict:
            self.config[key] = molecular_params_dict[key]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if os.name is not 'nt':
            self.config['path_to_data'] = os.path.abspath('/scratch/satrajit.chakrabarty/data/')
            self.config['excel_path'] = os.path.abspath("/home/satrajit.chakrabarty/molecular/molecular_status.xlsx")
        else:
            self.config['path_to_data'] = os.path.abspath('/sample_data/rcnn/')
            self.config['excel_path'] = os.path.abspath("/molecular/molecular_status.xlsx")

        self.config["training_modalities"] = ["T1c_subtrMeanDivStd", "T2_subtrMeanDivStd", "Flair_subtrMeanDivStd"]

        self.config["truth"] = ["OTMultiClass"]

        # self.config['dynamic_tr_val_steps'] = True
        # self.config['thickness'] = 2
        # self.config['slice_chooser'] = get_largest_tumor_slice
        # self.config["acs"] = 'sagittal'

        # Read excel
        df = pd.read_excel(self.config['excel_path'], sheet_name=self.config['worksheet_to_use'], engine='openpyxl')

        # Trim excel by removing entries containing Nans
        df = trim_df_by_dropping_nans(df, self.config)

        self.info['sessions_before_modality_omission'] = df['my_id'].tolist()

        # [Conditional] Trim cases based on presence in scratch and availability of all input modalities
        df = trim_df_based_on_presence_in_scratch_and_modality(df, self.config)

        self.info['sessions_after_modality_omission'] = df['my_id'].tolist()
        self.info['sessions_omitted_in_modality_omission'] = list(set(self.info['sessions_before_modality_omission']).difference(set(self.info['sessions_after_modality_omission'])))

        df = trim_df_based_on_GT(df, self.config, exclude_cases_with_partial_GT=False)
        df = trim_df_based_on_Tumor_modality(df, self.config)

        # ***************************************************************************************************************************
        # separate external cohort from different institution
        # internal testing cohort --> all cases from internal cohort that do not have segmentation gt
        # training/validation cohort --> all cases from internal cohort that have segmentation gt

        # Any additional conditions to filter data will come here
        df_int = df[~df.scratch_path.str.contains("MIRRIR_M19021_glioma") & ~df.scratch_path.str.contains("EGD")]  # internal cohort (to be split into train/val/test)
                
        df_train_val = df_int[df_int['seg_groundtruth'] == 'Yes']
        df_test = df_int[(df_int['seg_groundtruth'].isin(['No', 'partial']))]

        df_ext = df[df.scratch_path.str.contains("MIRRIR_M19021_glioma")]  # external cohort (to be used for external testing)
        df_EGD = df[df.scratch_path.str.contains("EGD")]  # external cohort (to be used for external testing)
        # End

        sessions_int = [os.path.abspath(os.path.join(self.config["path_to_data"], row['scratch_path'], row['my_id'])) for index, row in df_int.iterrows()]
        sessions_train_val = [os.path.abspath(os.path.join(self.config["path_to_data"], row['scratch_path'], row['my_id'])) for index, row in df_train_val.iterrows()]
        sessions_test = [os.path.abspath(os.path.join(self.config["path_to_data"], row['scratch_path'], row['my_id'])) for index, row in df_test.iterrows()]
        sessions_ext = [os.path.abspath(os.path.join(self.config["path_to_data"], row['scratch_path'], row['my_id'])) for index, row in df_ext.iterrows()]
        sessions_EGD = [os.path.abspath(os.path.join(self.config["path_to_data"], row['scratch_path'], row['my_id'])) for index, row in df_EGD.iterrows()]

        self.config['testing_sessions'] = sessions_test
        self.config['validation_sessions'] = []
        self.config['training_sessions'] = sessions_train_val
        self.config['ext_sessions'] = sessions_ext
        self.config['EGD_sessions'] = sessions_EGD

        self.info['class_distribution_overall'] = get_class_ids_from_df(df, self.config, self.info['sessions_after_modality_omission'])[0]
        self.info['class_distribution_int'] = str(get_class_ids_from_df(df, self.config, sessions_int)[0])        
        
        self.info['class_distribution_training'] = get_class_ids_from_df(df, self.config, self.config['training_sessions'])[0]
        df.at[df.index[df['my_id'].isin(get_basename_from_abspath(self.config['training_sessions']))].tolist(), 'cohort'] = 'train'
        
        if len(self.config['validation_sessions']):
            self.info['class_distribution_validation'] = str(get_class_ids_from_df(df, self.config, self.config['validation_sessions'])[0])
            df.at[df.index[df['my_id'].isin(get_basename_from_abspath(self.config['validation_sessions']))].tolist(),'cohort'] = 'val'

        if len(self.config['testing_sessions']):
            self.info['class_distribution_testing'] = get_class_ids_from_df(df, self.config, self.config['testing_sessions'])[0]
            df.at[df.index[df['my_id'].isin(get_basename_from_abspath(self.config['testing_sessions']))].tolist(), 'cohort'] = 'test'

        if len(self.config['ext_sessions']):
            self.info['class_distribution_ext'] = get_class_ids_from_df(df, self.config, self.config['ext_sessions'])[0]
            df.at[df.index[df['my_id'].isin(get_basename_from_abspath(self.config['ext_sessions']))].tolist(), 'cohort'] = 'ext'

        if len(self.config['EGD_sessions']): 
            self.info['class_distribution_EGD'] = get_class_ids_from_df(df, self.config, self.config['EGD_sessions'])[0]
            df.at[df.index[df['my_id'].isin(get_basename_from_abspath(self.config['EGD_sessions']))].tolist(),'cohort'] = 'EGD'

        
        self.df_filtered = df


        #########################
        #         I/O           #
        #########################

        self.model = model
        self.dim = dim
        # int [0 < dataset_size]. select n patients from dataset for prototyping.
        self.select_prototype_subset = None

        # some default paths.
        self.backbone_path = 'models/backbone.py'
        self.source_dir = os.path.dirname(os.path.realpath(__file__)) #current dir.
        self.model_path = 'models/{}.py'.format(self.model)

        

        self.input_df_name = 'info_df.pickle'
        self.pp_name = 'train'        
        self.pp_test_name = 'test'
        self.pp_WUSM_name = 'WUSM'
        self.pp_EGD_name = 'EGD'

        # path to preprocessed data.
        if os.name is not 'nt':
            self.root_dir = f"/scratch/satrajit.chakrabarty/mrcnn3d_mdt_datasets/{self.config['molecular_status']}_{self.dim}d/"
        else:
            self.root_dir = os.path.abspath(f"D:\\mrcnn3d_mdt\\datasets\\{self.config['molecular_status']}_{self.dim}d")

        self.pp_data_path = os.path.join(self.root_dir, self.pp_name)
        self.pp_test_data_path = os.path.join(self.root_dir, self.pp_test_name)
        self.pp_WUSM_data_path = os.path.join(self.root_dir, self.pp_WUSM_name)
        self.pp_EGD_data_path = os.path.join(self.root_dir, self.pp_EGD_name)     


        self.cohort_dict = {'train': {'path_to_data': self.pp_data_path},
                            'test': {'path_to_data': self.pp_test_data_path},
                            'WUSM': {'path_to_data': self.pp_WUSM_data_path},
                            'EGD': {'path_to_data': self.pp_EGD_data_path}
                            }  
  

        #########################
        #      Data Loader      #
        #########################

        #random seed for fold_generator and batch_generator.
        self.seed = 0

        #number of threads for multithreaded batch generation.
        n_workers = mp.cpu_count()
        self.n_workers = int(n_workers/2)

        # if True, segmentation losses learn all categories, else only foreground vs. background.
        self.class_specific_seg_flag = False


        #########################
        #      Architecture      #
        #########################

        self.weight_decay = 0.0

        # nonlinearity to be applied after convs with nonlinearity. one of 'relu' or 'leaky_relu'
        self.relu = 'relu'

        # if True initializes weights as specified in model script. else use default Pytorch init.
        self.custom_init = False

        # if True adds high-res decoder levels to feature pyramid: P1 + P0. (e.g. set to true in retina_unet configs)
        self.operate_stride1 = False

        #########################
        #  Schedule             #
        #########################

        # number of folds in cross validation.
        self.n_cv_splits = 5


        # number of probabilistic samples in validation.
        self.n_probabilistic_samples = None

        #########################
        #  Schedule / Selection #
        #########################

        self.num_epochs = 200
        self.batch_size = 32 if self.dim == 2 else 4
        self.num_train_batches = max(100, (len(self.config['training_sessions'])//self.batch_size)+1 )

        self.do_validation = False
        # decide whether to validate on entire patient volumes (like testing) or sampled patches (like training)
        # the former is morge accurate, while the latter is faster (depending on volume size)
        self.val_mode = 'val_patient' # one of 'val_sampling' , 'val_patient'
        
        if self.val_mode == 'val_patient':
            self.max_val_patients = None  # if 'None' iterates over entire val_set once.
        if self.val_mode == 'val_sampling':
            self.num_val_batches = 50

        #########################
        #   Testing / Plotting  #
        #########################

        # perform mirroring at test time. (only XY. Z not done to not blow up predictions times).
        self.test_aug = True

        # if True, test data lies in a separate folder and is not part of the cross validation.
        self.hold_out_test_set = True

        # if hold_out_test_set provided, ensemble predictions over models of all trained cv-folds.
        self.ensemble_folds = False

        # color specifications for all box_types in prediction_plot.
        self.box_color_palette = {'det': 'r', 'gt': 'g', 'neg_class': 'purple',
                              'prop': 'w', 'pos_class': 'g', 'pos_anchor': 'c', 'neg_anchor': 'c'}

        # scan over confidence score in evaluation to optimize it on the validation set.
        self.scan_det_thresh = False

        # plots roc-curves / prc-curves in evaluation.
        self.plot_stat_curves = False

        # evaluates average precision per image and averages over images. instead computing one ap over data set.
        self.per_patient_ap = False

        # threshold for clustering 2D box predictions to 3D Cubes. Overlap is computed in XY.
        self.merge_3D_iou = 0.1

        # monitor any value from training.
        self.n_monitoring_figures = 1
        # dict to assign specific plot_values to monitor_figures > 0. {1: ['class_loss'], 2: ['kl_loss', 'kl_sigmas']}
        self.assign_values_to_extra_figure = {}

        # save predictions to csv file in experiment dir.
        self.save_preds_to_csv = True

        # select a maximum number of patient cases to test. number or "all" for all
        self.max_test_patients = "all"


        # set the top-n-epochs to be saved for temporal averaging in testing.
        self.save_n_models = 3
        self.test_n_epochs = 1
        # set a minimum epoch number for saving in case of instabilities in the first phase of training.
        self.min_save_thresh = 0 if self.dim == 2 else 0

        self.report_score_level = ['patient', 'rois']  # choose list from 'patient', 'rois'
        self.class_assignment = {i: idx for idx, i in enumerate(self.config['labels_to_use'])}  # eg: {'codel': 1, 'non-codel': 0}, {'Mutant': 0, 'WT': 1} etc
        self.class_dict = {idx + 1: i for idx, i in enumerate(self.config['labels_to_use'])}  # 0 is background.
        self.patient_class_of_interest = 2  # patient metrics are only plotted for one class.
        self.ap_match_ious = [0.1]  # list of ious to be evaluated for ap-scoring.

        self.model_selection_criteria = [i + '_ap' for i in self.config['labels_to_use']]  # criteria to average over for saving epochs.

        self.plot_prediction_histograms = True
        self.plot_stat_curves = False

        #########################
        #   MRCNN               #
        #########################

        # if True, mask loss is not applied. used for data sets, where no pixel-wise annotations are provided.
        self.frcnn_mode = False

        # if True, unmolds masks in Mask R-CNN to full-res for plotting/monitoring.
        self.return_masks_in_val = True
        self.return_masks_in_test = True # needed if doing instance segmentation. evaluation not yet implemented.

        # add P6 to Feature Pyramid Network.
        self.sixth_pooling = False

        # for probabilistic detection
        self.n_latent_dims = 0


