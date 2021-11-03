import sys
import os
import glob
import torch
import torch.nn as nn
import logging.config
import argparse

from utils import *
from trainer import Trainer
from unet_architecture import UNet
from evaluator import evaluator_runner


def train_runner(params, outputpath, logger, mode='train'):
    """ Performs a single training or evaluation run

    Args:
        params (dataclass): class containing all parameters for training
        outputpath (str): path where training results are saved, for eval_only this is the
                        path where the trained models are evaluated
        logger : logger to write all messages to
        mode (str): can be 'eval_only' or 'train
    """

    # Define device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Data
    logger.info('Finding Files')

    # Get all training volumes
    trainfiles = glob.glob(os.path.join(params.trainset_path, '**',
                                        f'*{params.data_params.extension}'),
                           recursive=True)
    trainvolume_paths = [path for path in trainfiles if
                         params.data_params.volume_names in path]

    # Get all test volumes
    testfiles = glob.glob(os.path.join(
        params.testset_path, '**', f'*{params.data_params.extension}'),
        recursive=True)
    testvolume_paths = [
        path for path in testfiles if params.data_params.volume_names in path]

    # Load datasplit
    logger.info('Loading Datasplit')
    if mode == 'train':
        datasplitname = os.path.join(params.datasplit_path, params.data_params.datasplit_filename)
        datasplit = load_json(datasplitname)
 
    elif mode == 'eval_only':
        # Load dataset from experiment 
        datasplit = load_json(os.path.join(outputpath, 'datasplit.json'))

        # Update parameters based on experimental settings
        params = update_params(params, outputpath)

    # convert to full path names using extracted paths from train and test
    dataset = convert_datasplit(datasplit,
                                trainvolume_paths, testvolume_paths)

    # Print number of available scans
    for key in dataset.keys():
        logger.info('Total number of scans for ' +
                    key + ': ' + str(len(dataset[key])))

    # Build network and send to device
    net = UNet(1, len(params.structures) + 1, **vars(params.netw_params))
    net.to(device)

    # Distribute over multiple GPU's if desired
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    # Train model
    if mode == 'train':
        # Initialize trainer object
        trainer = Trainer(network=net, dataset=dataset,
                          device=device, savefolder=outputpath,
                          params=params)

        # Train model
        trainer.train_model()

        # Get path of best and last model
        bestmodelpath = trainer.best_model_path()
        last_checkpointpath = trainer.last_checkpoint_path()

    else:
        # path of best model
        bestmodelpath = os.path.join(outputpath, 'best_model_loss.tar')

        # obtain path from last checkpoint
        last_checkpointpath = get_lastcheckpoint_path(params.evalpath)

    logger.info('Start Evaluating data')

    # evaluate best model from validation loss
    logger.info('Start evaluating model with best validation loss')
    savepath_bestval = os.path.join(outputpath, 'model_bestval')
    saved_model = torch.load(bestmodelpath, map_location=device)
    net.load_state_dict(saved_model['model_state_dict'])
    evaluator_runner(net=net, params=params, dataset=dataset,
                     device=device, savepath=savepath_bestval,
                     save_images=params.save_params.save_predictions,
                     save_probs=params.save_params.save_probs)
    
    # evaluate last model from last checkpoint
    logger.info('Start evaluating last saved model')
    if last_checkpointpath is not None:
        savepath_last = os.path.join(outputpath, 'model_lastcheckpoint')
        saved_model = torch.load(last_checkpointpath, map_location=device)
        net.load_state_dict(saved_model['model_state_dict'])
        evaluator_runner(net=net, params=params, dataset=dataset,
                         device=device, savepath=savepath_last,
                         save_images=params.save_params.save_predictions,
                         save_probs=params.save_params.save_probs)


def setup_train_runner(args):
    """ Sets up everything for training runner

    Args:
        args (NameSpace): command line arguments
    """

    # Set up parameters from args
    params = params_setup(args)

    # Set up savelocation
    if not os.path.exists(params.savepath_exp):
        os.makedirs(params.savepath_exp)
    
    # Set logger
    set_logger(params.savepath_exp, comment=f'{params.mode}_')
    logger = logging.getLogger('__name__')


    # train new model from scratch
    if params.mode == 'train':
        for num in range(params.num_runs):
            # Make folder for run
            savefolder_run = os.path.join(params.savepath_exp, f'run_{num}')
            if not os.path.exists(savefolder_run):
                os.makedirs(savefolder_run)

            # Run Trainer
            logger.info(f'Starting run: {num+1} out of {params.num_runs}')
            try:
                train_runner(params, savefolder_run, logger, mode='train')
            except Exception:
                logger.error('Failed to run trainer', exc_info=True)
                sys.exit()

        # Summarize accross different runs
        if params.num_runs > 1:
            summarize_cv(params.savepath_exp, 'model_bestval')
            summarize_cv(params.savepath_exp, 'model_lastcheckpoint')

    # Evaluate all existing experiments in (nested) folders of path
    # Evaluates only model saved with best validation loss (best_model_loss.tar)
    elif params.mode == 'eval_only':

        evalpath = params.evalpath

        # Find all trained models in given folders
        eval_runs = glob.glob(os.path.join(evalpath, '**',  'best_model_loss.tar'),
                              recursive=True)
        
        for num, run in enumerate(eval_runs):
            # Get folder of model
            evalfolder = os.path.dirname(run)

            # Run evaluator
            logger.info(
                f'Starting evaluating: {num+1} out of {len(eval_runs)}')
            try:
                train_runner(params, evalfolder, logger, mode='eval_only')
            except Exception:
                logger.error('Failed to run evaluator', exc_info=True)
                sys.exit()


if __name__ == "__main__":
    """
    To run this code:
        - Create environment with 'conda env create -f config/environment.yml 
        (or install manual packages from requirements.txt)
        - Change parameters values in config/params.json
        - Change data and save paths in config/set_paths.py
    """
    parser = argparse.ArgumentParser(
        description='Run Subcortical Segmentation Training Code')

    parser.add_argument('-p', '--parameter_path', action='store', default='config/params.json',
                        help='Define the path to the parameter file used for training')
    parser.add_argument('-s', '--savepath', type=str, 
                        help='Define the folder for saving everything, inside this folder a new folder is made with same name of the parameterfile')
    parser.add_argument('-d', '--datapath', type=str, 
                        help='Define the base folder of the data, e.g. /path/to/Data')
    parser.add_argument('-e', '--evalpath', type= str)

    args = parser.parse_args()

    setup_train_runner(args)
