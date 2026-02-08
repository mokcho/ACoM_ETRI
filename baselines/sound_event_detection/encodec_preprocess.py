# -*- coding: utf-8 -*-
import argparse
import datetime
import inspect
import os
import time
from pprint import pprint

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from data_utils.Desed import DESED
from data_utils.DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from TestModel import _load_crnn, _load_state_vars, get_variables
from evaluation_measures import get_predictions, psds_score, compute_psds_from_operating_points, compute_metrics
from models.CRNN import CRNN
import config as cfg
from utilities import ramps
from utilities.Logger import create_logger
from utilities.Scaler import ScalerPerAudio, Scaler
from utilities.utils import SaveBest, to_cuda_if_available, weights_init, AverageMeterSet, EarlyStopping, \
    get_durations_df
from utilities.ManyHotEncoder import ManyHotEncoder
from utilities.Transforms import get_transforms


def adjust_learning_rate(optimizer, rampup_value, rampdown_value=1):
    """ adjust the learning rate
    Args:
        optimizer: torch.Module, the optimizer to be updated
        rampup_value: float, the float value between 0 and 1 that should increases linearly
        rampdown_value: float, the float between 1 and 0 that should decrease linearly
    Returns:

    """
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    # We commented parts on betas and weight decay to match 2nd system of last year from Orange
    lr = rampup_value * rampdown_value * cfg.max_learning_rate
    # beta1 = rampdown_value * cfg.beta1_before_rampdown + (1. - rampdown_value) * cfg.beta1_after_rampdown
    # beta2 = (1. - rampup_value) * cfg.beta2_during_rampdup + rampup_value * cfg.beta2_after_rampup
    # weight_decay = (1 - rampup_value) * cfg.weight_decay_during_rampup + cfg.weight_decay_after_rampup * rampup_value

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # param_group['betas'] = (beta1, beta2)
        # param_group['weight_decay'] = weight_decay


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_params, params in zip(ema_model.parameters(), model.parameters()):
        ema_params.data.mul_(alpha).add_(1 - alpha, params.data)


def train(train_loader, model, optimizer, c_epoch, ema_model=None, mask_weak=None, mask_strong=None, adjust_lr=False):
    """ One epoch of a Mean Teacher model
    Args:
        train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
            Should return a tuple: ((teacher input, student input), labels)
        model: torch.Module, model to be trained, should return a weak and strong prediction
        optimizer: torch.Module, optimizer used to train the model
        c_epoch: int, the current epoch of training
        ema_model: torch.Module, student model, should return a weak and strong prediction
        mask_weak: slice or list, mask the batch to get only the weak labeled data (used to calculate the loss)
        mask_strong: slice or list, mask the batch to get only the strong labeled data (used to calcultate the loss)
        adjust_lr: bool, Whether or not to adjust the learning rate during training (params in config)
    """
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    class_criterion = nn.BCELoss()
    consistency_criterion = nn.MSELoss()
    class_criterion, consistency_criterion = to_cuda_if_available(class_criterion, consistency_criterion)

    meters = AverageMeterSet()
    # log.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    for i in train_loader:
        print(i)
        break
    for i, ((batch_input, ema_batch_input), target) in enumerate(train_loader):
        global_step = c_epoch * len(train_loader) + i
        rampup_value = ramps.exp_rampup(global_step, cfg.n_epoch_rampup*len(train_loader))

        if adjust_lr:
            adjust_learning_rate(optimizer, rampup_value)
        meters.update('lr', optimizer.param_groups[0]['lr'])
        batch_input, ema_batch_input, target = to_cuda_if_available(batch_input, ema_batch_input, target)
        # Outputs
        strong_pred_ema, weak_pred_ema = ema_model(ema_batch_input)
        strong_pred_ema = strong_pred_ema.detach()
        weak_pred_ema = weak_pred_ema.detach()
        strong_pred, weak_pred = model(batch_input)

        loss = None
        # Weak BCE Loss
        target_weak = target.max(-2)[0]  # Take the max in the time axis
        if mask_weak is not None:
            weak_class_loss = class_criterion(weak_pred[mask_weak], target_weak[mask_weak])
            ema_class_loss = class_criterion(weak_pred_ema[mask_weak], target_weak[mask_weak])
            loss = weak_class_loss

            if i == 0:
                log.debug(f"target: {target.mean(-2)} \n Target_weak: {target_weak} \n "
                          f"Target weak mask: {target_weak[mask_weak]} \n "
                          f"Target strong mask: {target[mask_strong].sum(-2)}\n"
                          f"weak loss: {weak_class_loss} \t rampup_value: {rampup_value}"
                          f"tensor mean: {batch_input.mean()}")
            meters.update('weak_class_loss', weak_class_loss.item())
            meters.update('Weak EMA loss', ema_class_loss.item())

        # Strong BCE loss
        if mask_strong is not None:
            strong_class_loss = class_criterion(strong_pred[mask_strong], target[mask_strong])
            meters.update('Strong loss', strong_class_loss.item())

            strong_ema_class_loss = class_criterion(strong_pred_ema[mask_strong], target[mask_strong])
            meters.update('Strong EMA loss', strong_ema_class_loss.item())

            if loss is not None:
                loss += strong_class_loss
            else:
                loss = strong_class_loss

        # Teacher-student consistency cost
        if ema_model is not None:
            consistency_cost = cfg.max_consistency_cost * rampup_value
            meters.update('Consistency weight', consistency_cost)
            # Take consistency about strong predictions (all data)
            consistency_loss_strong = consistency_cost * consistency_criterion(strong_pred, strong_pred_ema)
            meters.update('Consistency strong', consistency_loss_strong.item())
            if loss is not None:
                loss += consistency_loss_strong
            else:
                loss = consistency_loss_strong
            meters.update('Consistency weight', consistency_cost)
            # Take consistency about weak predictions (all data)
            consistency_loss_weak = consistency_cost * consistency_criterion(weak_pred, weak_pred_ema)
            meters.update('Consistency weak', consistency_loss_weak.item())
            if loss is not None:
                loss += consistency_loss_weak
            else:
                loss = consistency_loss_weak

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'
        meters.update('Loss', loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if ema_model is not None:
            update_ema_variables(model, ema_model, 0.999, global_step)

    epoch_time = time.time() - start
    log.info(f"Epoch: {c_epoch}\t Time {epoch_time:.2f}\t {meters}")
    return loss

def train_model(train_loader, model, optimizer, c_epoch, mask_weak=None, mask_strong=None, adjust_lr=False):
    """ One epoch of training without EMA """
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    class_criterion = nn.BCELoss()
    class_criterion = to_cuda_if_available(class_criterion)

    meters = AverageMeterSet()
    start = time.time()

    for i, ((batch_input), target) in enumerate(train_loader):  # Ignore EMA input
        global_step = c_epoch * len(train_loader) + i
        rampup_value = ramps.exp_rampup(global_step, cfg.n_epoch_rampup * len(train_loader))

        if adjust_lr:
            adjust_learning_rate(optimizer, rampup_value)
        meters.update('lr', optimizer.param_groups[0]['lr'])

        batch_input, target = to_cuda_if_available(batch_input, target)
        strong_pred, weak_pred = model(batch_input)

        loss = None
        target_weak = target.max(-2)[0]  # collapse time

        if mask_weak is not None:
            weak_class_loss = class_criterion(weak_pred[mask_weak], target_weak[mask_weak])
            meters.update('weak_class_loss', weak_class_loss.item())
            loss = weak_class_loss

        if mask_strong is not None:
            strong_class_loss = class_criterion(strong_pred[mask_strong], target[mask_strong])
            meters.update('strong_class_loss', strong_class_loss.item())
            loss = loss + strong_class_loss if loss is not None else strong_class_loss

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), f'Loss explosion: {loss.item()}'
        assert loss.item() >= 0, 'Loss should not be negative'

        meters.update('loss', loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log.info(f"Epoch: {c_epoch}\t Time: {time.time() - start:.2f}s\t {meters}")
    return loss


def get_dfs(desed_dataset, nb_files=None, separated_sources=False):
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    audio_weak_ss = None
    audio_unlabel_ss = None
    audio_validation_ss = None
    audio_synthetic_ss = None
    if separated_sources:
        audio_weak_ss = cfg.weak_ss
        audio_unlabel_ss = cfg.unlabel_ss
        audio_validation_ss = cfg.validation_ss
        audio_synthetic_ss = cfg.synthetic_ss

    weak_df = desed_dataset.initialize_and_get_df(cfg.weak, audio_dir_ss=audio_weak_ss, nb_files=nb_files)
    unlabel_df = desed_dataset.initialize_and_get_df(cfg.unlabel, audio_dir_ss=audio_unlabel_ss, nb_files=nb_files, download=False)
    # Event if synthetic not used for training, used on validation purpose
    # synthetic_df = desed_dataset.initialize_and_get_df(cfg.synthetic, audio_dir_ss=audio_synthetic_ss,
                                                    #    nb_files=nb_files, download=False)
    # log.debug(f"synthetic: {synthetic_df.head()}")
    validation_df = desed_dataset.initialize_and_get_df(cfg.validation, audio_dir=cfg.audio_validation_dir,
                                                        audio_dir_ss=audio_validation_ss, nb_files=nb_files)
    # Divide synthetic in train and valid
    # filenames_train = synthetic_df.filename.drop_duplicates().sample(frac=0.8, random_state=26)
    # train_synth_df = synthetic_df[synthetic_df.filename.isin(filenames_train)]
    # valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)
    # Put train_synth in frames so many_hot_encoder can work.
    #  Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    # train_synth_df.onset = train_synth_df.onset * cfg.sample_rate // cfg.hop_size // pooling_time_ratio
    # train_synth_df.offset = train_synth_df.offset * cfg.sample_rate // cfg.hop_size // pooling_time_ratio
    # log.debug(valid_synth_df.event_label.value_counts())

    data_dfs = {"weak": weak_df,
                "unlabel": unlabel_df,
                # "synthetic": synthetic_df,
                # "train_synthetic": train_synth_df,
                # "valid_synthetic": valid_synth_df,
                "validation": validation_df,
                }

    return data_dfs


if __name__ == '__main__':
    torch.manual_seed(2020)
    np.random.seed(2020)
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    logger.info("Baseline 2020")
    logger.info(f"Starting time: {datetime.datetime.now()}")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", '--model_path', type=str, default='/home/work/jjk/data/dcase_ckpt/dcase20_baseline_model.p',
                        help="Path of the model to be evaluated")
    
    
    parser.add_argument("-g", '--groundtruth_tsv', type=str, default='/home/work/jjk/data/audioset/audioset/metadata/validation/validation.tsv',
                        help="Path of the groundtruth tsv file")
    
    parser.add_argument("-mw", "--median_window", type=int, default=None,
                        help="Nb of frames for the median window, "
                             "if None the one defined for testing after training is used")

    parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                        help="Number of files to be used. Useful when testing on small number of files.")
    parser.add_argument('--meta_gt', type=str, default=None,
                        help="Path of the groundtruth description of feat_filenames and durations")
    parser.add_argument("-ga", '--groundtruth_audio_dir', type=str, default='/home/work/jjk/data/audioset/audioset/audio/',
                        help="Path of the groundtruth filename, (see in config, at dataset folder)")
    
    parser.add_argument("-n", '--no_synthetic', dest='no_synthetic', action='store_true', default=True,
                        help="Not using synthetic labels during training")
    parser.add_argument('--nb_files', type=int, default=None,
                        help="Number of files to be used. Useful when testing on small number of files.")

    parser.add_argument("-br", "--bitrate", type=float, default=12.0, help="bitrate for EnCodec")
    
    f_args = parser.parse_args()
    pprint(vars(f_args))
    
    model_path, median_window, gt_audio_dir, groundtruth, durations = get_variables(f_args)

    reduced_number_of_data = f_args.subpart_data
    no_synthetic = f_args.no_synthetic

    if no_synthetic:
        add_dir_model_name = "_no_synthetic"
    else:
        add_dir_model_name = "_with_synthetic"

    store_dir = os.path.join("stored_data", "MeanTeacher" + add_dir_model_name)
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")
    os.makedirs(store_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)
    os.makedirs(saved_pred_dir, exist_ok=True)

    n_channel = 1
    add_axis_conv = 0

    # Model taken from 2nd of dcase19 challenge: see Delphin-Poulat2019 in the results.
    n_layers = 7
    crnn_kwargs = {"n_in_channel": n_channel, "nclass": len(cfg.classes), "attention": True, "n_RNN_cell": 128,
                   "n_layers_RNN": 2,
                   "activation": "glu",
                   "dropout": 0.5,
                   "kernel_size": n_layers * [3], "padding": n_layers * [1], "stride": n_layers * [1],
                   "nb_filters": [16,  32,  64,  128,  128, 128, 128],
                   "pooling": [[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]}
    pooling_time_ratio = 4  # 2 * 2

    out_nb_frames_1s = cfg.sample_rate / cfg.hop_size / pooling_time_ratio
    median_window = max(int(cfg.median_window_s * out_nb_frames_1s), 1)
    logger.debug(f"median_window: {median_window}")
    # ##############
    # DATA
    # ##############
    
    dataset = DESED(base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"), compute_log=False, use_encodec=True, bitrate=f_args.bitrate)
    dfs = get_dfs(dataset, reduced_number_of_data)

    # Meta path for psds
    durations_synth = get_durations_df(cfg.synthetic)
    many_hot_encoder = ManyHotEncoder(cfg.classes, n_frames=cfg.max_frames // pooling_time_ratio)
    encod_func = many_hot_encoder.encode_strong_df

    # Normalisation per audio or on the full dataset
    if cfg.scaler_type == "dataset":
        transforms = get_transforms(cfg.max_frames, add_axis=add_axis_conv)
        weak_data = DataLoadDf(dfs["weak"], encod_func, transforms)
        unlabel_data = DataLoadDf(dfs["unlabel"], encod_func, transforms)
        # train_synth_data = DataLoadDf(dfs["train_synthetic"], encod_func, transforms)
        scaler_args = []
        scaler = Scaler()
        # # Only on real data since that's our final goal and test data are real
        scaler.calculate_scaler(ConcatDataset([weak_data, unlabel_data])) #, train_synth_data]))
        # scaler.calculate_scaler(weak_data)
        logger.debug(f"scaler mean: {scaler.mean_}")
    else:
        scaler_args = ["global", "min-max"]
        scaler = ScalerPerAudio(*scaler_args)

    transforms = get_transforms(cfg.max_frames, scaler, add_axis_conv,
                                noise_dict_params={"mean": 0., "snr": cfg.noise_snr})
    transforms_valid = get_transforms(cfg.max_frames, scaler, add_axis_conv)

    weak_data = DataLoadDf(dfs["weak"], encod_func, transforms, in_memory=cfg.in_memory)
    unlabel_data = DataLoadDf(dfs["unlabel"], encod_func, transforms, in_memory=cfg.in_memory_unlab)