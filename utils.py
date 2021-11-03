import os
import json
import logging
import SimpleITK as sitk
import glob
import pandas as pd
import torch.nn.functional as F
from define_parameters import Parameters
from pathlib import Path


def params_setup(args):
    """ Combine parameters from command line arguments and json
    dictionary with settings

    Args:
        args (Namespace): Namespace attribute containing
        'parameter_path', 'savepath' and 'datapath'
    Returns:
        Dataclass : Class with all parameters
    """

    assert os.path.isfile(args.parameter_path), \
        "No json configuration file found at {}".format(
        args.parameter_path)

    param_dict = load_json(args.parameter_path)

    # Add parameters from args to dictionary
    args_dict = vars(args)
    for key in args_dict.keys():
        if args_dict[key] is not None:
            param_dict[key] = args_dict[key]
    param_dict['exp_name'] = os.path.basename(args.parameter_path)

    # Make parameter object
    params = Parameters.from_dict(param_dict)

    return params


def update_params(params, path):
    """ Updates parameters from previous experiment with correct data and savepath (only used for evaluation only mode)

    Args:
        params (Dataclass): parameters from eval run
        path (str): path to the model to be evaluated

    Returns:
        [Dataclass]: updated parameters
    """

    exp_params = load_json(os.path.join(path, 'config.json'))

    exp_params['datapath'] = params.datapath
    exp_params['savepath'] = path
    exp_params['evalpath'] = path

    exp_params['train_batchsize'] = params.train_batchsize
    exp_params['val_batchsize'] = params.val_batchsize

    params = Parameters.from_dict(exp_params)

    return params


def load_json(json_path):
    with open(json_path) as f:
        dict = json.load(f)
    return dict


def save_json(dump_dict, json_path):
    with open(json_path, 'w') as f:
        json.dump(dump_dict, f)


def set_logger(log_path, comment=''):
    """ Set up the logger 

    Args:
        log_path (str):  location to save logger
        comment (str, optional): Additional comment to add to savename. Defaults to ''.
    """

    savename = comment + 'logfile.log'

    config = {'version': 1,
              'formatters': {
                  'default': {
                      'datefmt': "%Y-%m-%d %H:%M:%S",
                      'format': '%(asctime)s %(levelname)s : %(filename)s : Line %(lineno)s : %(funcName)20s() : %(message)s',
                  },
              },
              'handlers': {
                  'console': {
                      'class': "logging.StreamHandler",
                      'formatter': 'default'
                  },
                  'error_file': {
                      'class': "logging.FileHandler",
                      'mode': 'a',
                      'formatter': 'default',
                      'filename': os.path.join(log_path, savename),
                  },
              },
              'loggers': {
                  '': {
                      'handlers': ['error_file', 'console'],
                      'level': 'INFO',
                  },
              },
              }

    logging.config.dictConfig(config)


def saveas_MHA_fun(im_array, save_path, spacing=(0.6, 0.6, 0.6), origin=(0, 0, 0)):
    """ Saves a numpy array as image

    Args:
        im_array (array): image array to save
        save_path (str): path where image is saved
        spacing (tuple, optional): Spacing of the image. Defaults to (0.6, 0.6, 0.6).
        origin (tuple, optional): Origin of the image. Defaults to (0, 0, 0).
    """
    im = sitk.GetImageFromArray(im_array)
    im.SetSpacing(spacing)
    im.SetOrigin(origin)

    if os.path.isdir(os.path.dirname(save_path)) is not True:
        os.makedirs(os.path.dirname(save_path))

    sitk.WriteImage(im, save_path)


def convert_datasplit(datasplit, filenames, filenames_test):
    """ Takes in the datasplit (only names of files) and all the full paths to the data

    Args:
        datasplit (dict): dict with for every key a list of filenames
        filenames (list):  list of full paths to traindata
        filenames_test (list): list of full paths to test data

    Returns:
        dict: same as datasplit but now with full filepaths 
    """

    newdict = {}

    for datapart in datasplit.keys():

        subdata = datasplit[datapart]

        # Remove file extensions from datasplit (if any)
        subdata_ext = [Path(x).stem for x in subdata]

        # For test, take filenames corresponding to manual labels
        if datapart == 'test':
            fullfiles = [x for x in filenames_test if
                         Path(x).stem in subdata_ext]
        else:
            fullfiles = [x for x in filenames if
                         Path(x).stem in subdata_ext]

        newdict[datapart] = fullfiles
    return newdict


def summarize_cv(eval_path, modeltype):
    """ Summarize results of multiple runs

    Args:
        eval_path (str): path to summarize results from
        modeltype (str): name of folder inside experiment
    """

    if os.path.isdir(os.path.join(eval_path, 'run_0')):
        folds = glob.glob(os.path.join(eval_path, 'run*', modeltype))
    else:
        folds = [os.path.join(eval_path, modeltype)]

    datatypes = glob.glob(os.path.join(folds[0], '*evalmetrics.xlsx'))

    for data in datatypes:
        dataname = os.path.basename(data)

        # get number of sheets
        results = pd.ExcelFile(os.path.join(folds[0], dataname))
        metrics = results.sheet_names

        if not os.path.exists(os.path.join(eval_path, modeltype)):
            os.makedirs(os.path.join(eval_path, modeltype))

        writer = pd.ExcelWriter(os.path.join(
            eval_path, modeltype, dataname.replace('.xlsx', '_average.xlsx')),
            engine='openpyxl')

        for metric in metrics:
            av_list = []
            std_list = []
            for fold in folds:
                results = pd.read_excel(
                    os.path.join(fold, dataname), sheet_name=metric,
                    index_col=0)

                # get mean and std
                average_perform = results.loc['mean'][1:].to_dict()
                std_perform = results.loc['std'][1:].to_dict()

                av_list.append(average_perform)
                std_list.append(std_perform)

            std_df = pd.DataFrame(std_list)
            av_df = pd.DataFrame(av_list)

            std_df.loc['mean'] = std_df.mean()
            av_df.loc['mean'] = av_df.mean()

            combined = pd.concat([av_df, std_df], keys=['average', 'std'])
            combined.to_excel(
                writer, sheet_name=metric, float_format='%.3f')
        writer.save()
        writer.close()



def convert_one_hot(target_torch, num_classes):
    """ Converts multi-label to one-hot
    Args:
        target_torch (tensor):  tensor with labels in one hot format
    """
    # transform targets to one-hot
    one_hot = F.one_hot(target_torch[0].long(),
                        num_classes=num_classes)
    one_hot = one_hot.permute(3, 0, 1, 2)
    return one_hot

    

def get_lastcheckpoint_path(folder, return_epoch=False):
    """ Searches given folder for the last checkpoint

    Args:
        folder (str):  path of folder to search in
        return_epoch (bool, optional): Whether to return epoch number. Defaults to False.

    Returns:
        str (, int) : path of last checkpoint (, number of last epoch (optional))
    """
    checkpoints = glob.glob(os.path.join(folder, '*checkpoint*.tar'))
    checkpoints_names = [os.path.basename(x) for x in checkpoints]
    epochs = [int(x.split('_')[2])
              for x in checkpoints_names]

    if len(epochs) > 0:
        index = epochs.index(max(epochs))
        last_checkpointpath = checkpoints[index]
        if return_epoch:
            return last_checkpointpath, max(epochs)
        else:
            return last_checkpointpath
    else:
        return None


