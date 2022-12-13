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

"""execution script."""

import argparse
import os
import time
import torch

import utils.exp_utils as utils
from utils import mrcnn_utils
from utils import config_utils
from evaluator import Evaluator
from predictor import Predictor
from plotting import plot_batch_prediction
import pprint
import glob
import visualize
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import importlib

import matplotlib.pyplot as plt
from data_scripts import write_largest_slice_to_excel as find_largest_slice
from torchinfo import summary

def train(logger):
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """
    logger.info('performing training in {}D over fold {} on experiment {} with model {}'.format(
        cf.dim, cf.fold, cf.exp_dir, cf.model))

    net = model.net(cf, logger).cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay)
    model_selector = utils.ModelSelector(cf, logger)
    train_evaluator = Evaluator(cf, logger, mode='train')
    val_evaluator = Evaluator(cf, logger, mode=cf.val_mode)

    starting_epoch = 1

    # prepare monitoring
    monitor_metrics, TrainingPlot = utils.prepare_monitoring(cf)

    if cf.resume_to_checkpoint:
        starting_epoch, monitor_metrics = utils.load_checkpoint(cf.resume_to_checkpoint, net, optimizer)
        logger.info('resumed to checkpoint {} at epoch {}'.format(cf.resume_to_checkpoint, starting_epoch))

    logger.info('loading dataset and initializing batch generators...')
    batch_gen = data_loader.get_train_generators(cf, logger)

    for epoch in range(starting_epoch, cf.num_epochs + 1):

        logger.info('starting training epoch {}'.format(epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cf.learning_rate[epoch - 1]

        start_time = time.time()

        net.train()
        train_results_list = []

        for bix in range(cf.num_train_batches):
            batch = next(batch_gen['train'])
            tic_fw = time.time()
            results_dict = net.train_forward(batch)
            tic_bw = time.time()
            optimizer.zero_grad()
            results_dict['torch_loss'].backward()
            optimizer.step()
            logger.info('tr. batch {0}/{1} (ep. {2}/{4}) total {3:.3f}s || '.format(bix + 1, cf.num_train_batches, epoch, time.time() - tic_fw, cf.num_epochs) + results_dict['logger_string'])
            train_results_list.append([results_dict['boxes'], batch['pid']])
            monitor_metrics['train']['monitor_values'][epoch].append(results_dict['monitor_values'])

        _, monitor_metrics['train'] = train_evaluator.evaluate_predictions(train_results_list, monitor_metrics['train'])
        train_time = time.time() - start_time

        with torch.no_grad():
            net.eval()
            if cf.do_validation:

                logger.info('starting validation in mode {}.'.format(cf.val_mode))

                val_predictor = Predictor(cf, net, logger, mode='val')
                # val_results_list = []
                # for _ in range(batch_gen['n_val']):
                #     batch = next(batch_gen[cf.val_mode])
                #     if cf.val_mode == 'val_patient':
                #         results_dict = val_predictor.predict_patient(batch)
                #     elif cf.val_mode == 'val_sampling':
                #         results_dict = net.train_forward(batch, is_validation=True)
                #     val_results_list.append([results_dict['boxes'], batch['pid']])
                #     monitor_metrics['val']['monitor_values'][epoch].append(results_dict['monitor_values'])

                # _, monitor_metrics['val'] = val_evaluator.evaluate_predictions(val_results_list, monitor_metrics['val'])
                

                batch = next(batch_gen['val_patient'])
                results_dict = val_predictor.predict_patient(batch)

                logger.info('plotting predictions from validation sampling.')
                plot_batch_prediction(batch, results_dict, cf)

        # Saving model checkpoint based on metrics
        model_selector.run_model_selection(net, optimizer, monitor_metrics, epoch)                

        # update monitoring and prediction plots
        TrainingPlot.update_and_save(monitor_metrics, epoch)
        epoch_time = time.time() - start_time
        logger.info('trained epoch {}: took {:.2f} sec. ({:.2f} train / {:.2f} val)'.format(epoch, epoch_time, train_time, epoch_time-train_time))


def test(logger, cohort_suffix, model_suffix = 'last'):
    """
    perform testing for a given fold (or hold out set). save stats in evaluator.
    """
    path_to_data = cf.cohort_dict[cohort_suffix]['path_to_data']

    logger.info('starting testing model of fold {} in exp {}'.format(cf.fold, cf.exp_dir))
    
    # define model
    net = model.net(cf, logger).cuda()
    
    # define generator
    batch_gen = data_loader.get_test_generator(cf, logger, path_to_data)

    # define prediction (model weights are loaded here)
    test_predictor = Predictor(cf, net, logger, mode='test')
    test_results_list = test_predictor.predict_test_set(batch_gen, cohort_suffix, return_results=True, model_suffix = model_suffix)

    # logger.debug("test_results_list")
    # pprint.pprint(test_results_list)

    # define evaluator
    test_evaluator = Evaluator(cf, logger, mode='test')    
    test_evaluator.evaluate_predictions(test_results_list)
    test_evaluator.score_test_df()


def inspect_data(cf, path_to_cohort_data):

    path_to_inspect_data = os.path.join(path_to_cohort_data, 'inspect_data')
    if not os.path.exists(path_to_inspect_data): os.makedirs(path_to_inspect_data)

    p_df = pd.read_pickle(os.path.join(path_to_cohort_data, cf.input_df_name))

    class_targets = p_df['class_id'].tolist()
    pids = p_df.pid.tolist()
    path_to_all_data = [os.path.join(path_to_cohort_data, '{}.npy'.format(pid)) for pid in pids]

    for i, pid in zip(path_to_all_data, pids):

        all_data = np.load(i, mmap_mode='r')
        image = all_data[:-1] # nb_channels x 128 x 128
        mask = all_data[-1].astype('uint8') # 128 x 128
        mask = np.expand_dims(mask, axis=-1) # 128 x 128 x 1

        class_ids = np.array([p_df[p_df['pid'] == pid]['class_id'].iloc[0]])
        session_name = p_df[p_df['pid'] == pid]['subj_id'].iloc[0]

        # print('\r[{}/{}] image inspected: {:.1f}%, {}'.format(count, len(dataset_cohort.image_ids), count / float(len(dataset_cohort.image_ids)) * 100, session_name))

        # Compute Bounding box
        bbox = mrcnn_utils.extract_bboxes(mask)
        print(os.path.join(path_to_inspect_data, "train+" + session_name + ".png"))
        visualize.display_instances_all_mods(image,
                                             bbox,
                                             mask,
                                             class_ids,
                                             cf.config['labels_to_use'],
                                             savepath=os.path.join(path_to_inspect_data, "train+" + session_name + ".png"),
                                             suptitle = config_utils.get_suptitle(cf.df_filtered, session_name, cf.config['fold'], cf.config))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--mol_stat', type=str, default='idh', help='currently only supports "idh"')
    parser.add_argument('-d', '--dim', type=str, default='3', help='currently only supports "3" i.e., 3d')
    parser.add_argument('-e', '--exp', type=str, default='001', help='Given exp id will be used to choose config from config_files/idh/')

    parser.add_argument('-m', '--mode', type=str, default='train', help='one out of: train/test ')
    parser.add_argument('-u', '--use_stored_settings', default=False, action='store_true', help='load configs from existing exp_dir instead of source dir.')
    parser.add_argument('--resume_to_checkpoint', type=str, default=None, help='if resuming to checkpoint, the desired fold still needs to be parsed via --folds.')
    parser.add_argument('--dev', default=False, action='store_true', help="development mode: shorten everything")
    parser.add_argument('--inspect', default=False, action='store_true', help="inspect training and test data")

    args = parser.parse_args()

    resume_to_checkpoint = args.resume_to_checkpoint
    folds = [0]
    fold = folds[0]


    exp_source = os.path.join(os.path.dirname(__file__), 'config_files', f"{str(args.mol_stat)}") # config_files/idh
    cfg_name = f"{args.exp}.py"
    exp_op_dir = os.path.join('/scratch/satrajit.chakrabarty/mrcnn3d_mdt_experiments', f"{str(args.mol_stat)}_{str(args.dim)}d_{args.exp}") # eg: idh_2d_001
    
    print("exp_source = ", exp_source)
    print("cfg_name = ", cfg_name)
    print("exp_op_dir = ", exp_op_dir)

    # set up experiment and logger
    if 'train' in args.mode:
        cf = utils.prep_exp(exp_source, exp_op_dir, cfg_name, args.dim, is_training=True, use_stored_settings=False)
    else:
        cf = utils.prep_exp(exp_source, exp_op_dir, cfg_name, args.dim, is_training=False, use_stored_settings=True)

    logger = utils.get_logger(cf.exp_dir)
    cf.folds = folds
    cf.fold = folds[0]
    cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
    if not os.path.exists(cf.fold_dir): os.mkdir(cf.fold_dir)
    cf.data_dest = None

    # assert that input molecular status and molecular status specified in config are same
    assert args.mol_stat.lower() == cf.config['molecular_status'].lower(), \
        f"Input molecular status = {args.mol_stat}, but config has = {cf.config['molecular_status']}"


    overwrite = False

    
    if args.mode == 'train':

        #########################
        #         Train         #
        #########################

        # check if data required for the experiment is available
        data_gen = importlib.import_module(f'data_scripts.generate_npy_from_nifti_{str(args.dim)}d')

        

        if overwrite or (not os.path.exists(cf.pp_data_path)) or (len(glob.glob(os.path.join(cf.pp_data_path, "*.npy"))) == 0):
            data_gen.main(args, cohort_suffix = 'train', print_fn=logger.info)

        if overwrite or (not os.path.exists(cf.pp_test_data_path)) or (len(glob.glob(os.path.join(cf.pp_test_data_path, "*.npy"))) == 0):
            data_gen.main(args, cohort_suffix = 'test', print_fn=logger.info)

        # Inspect data (2d)
        if (args.inspect) and (args.dim is not "3"):
            inspect_data(cf, path_to_cohort_data = cf.pp_data_path)
            inspect_data(cf, path_to_cohort_data = cf.pp_test_data_path)


        # write df_filtered.csv to exp_op_dir
        cf.df_filtered.to_csv(os.path.join(exp_op_dir, 'df_filtered.csv'), index=False)

        if args.dev:
            cf.num_epochs = 5
            cf.num_train_batches = 5
            cf.num_val_batches = 5
            cf.batch_size = 5
            cf.max_val_patients = 5


        model = utils.import_module('model', cf.model_path)
        data_loader = utils.import_module('dl', os.path.join('data_scripts', 'data_loader.py'))

        
        cf.resume_to_checkpoint = resume_to_checkpoint
        

        config_utils.print_data_info(logger, cf, cf.config, cf.info)
        train(logger)
        cf.resume_to_checkpoint = None


    elif args.mode == 'test':

        if args.dev:
            cf.batch_size = 3
            cf.max_test_patients = 20


        model = utils.import_module('model', cf.model_path)
        data_loader = utils.import_module('dl', os.path.join('data_scripts', 'data_loader.py'))


        # check if data required for the experiment is available
        data_gen = importlib.import_module(f'data_scripts.generate_npy_from_nifti_{str(args.dim)}d')

        model_suffix = 'last'

        for cohort_suffix in ['test', 'WUSM']:

            if overwrite or (not os.path.exists(cf.cohort_dict[cohort_suffix]['path_to_data'])) or (len(glob.glob(os.path.join(cf.cohort_dict[cohort_suffix]['path_to_data'], "*.npy"))) == 0):
                data_gen.main(args, cohort_suffix = cohort_suffix, print_fn=logger.info)   
            
            if not os.path.exists(os.path.join(cf.exp_dir, f'results_{model_suffix}+{cohort_suffix}.csv')):
                test(logger, cohort_suffix, model_suffix)
        
                predictor = Predictor(cf, net=None, logger=logger, mode='analysis')
                results_list = predictor.load_saved_predictions(cohort_suffix, model_suffix, apply_wbc=True)
                utils.create_csv_output(results_list, cohort_suffix, model_suffix, cf, logger)

            config_utils.get_test_csv_from_result_csv(cf, cohort_suffix, model_suffix, exp_op_dir)
            config_utils.plot_ROC_PR_CM_from_df(cf, exp_op_dir, cohort_suffix, model_suffix)