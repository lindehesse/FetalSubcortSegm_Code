import numpy as np
import os
import pandas as pd
import logging
import torch
import torch.nn.functional as F
from datagenerator import MyDataset_Preloader, MyManualTestset
from lossfile import calc_dice
from utils import saveas_MHA_fun
from evaluation_metrics import get_surface_metrics


class Evaluator():
    def __init__(self, net, params, dataset, device, savepath, savename,
                 save_images=True, save_probs=False):
        """ Evaluator class to perform final evaluation of trained model

        Args:
            net (network): network to evaluated
            params (Dataclass):  Parameter object with all settings
            dataset (list): list of filenames to evaluate
            device (Torch device): device where to evaluate
            savepath (str): path where results are saved
            savename (str): name to add to savepath
            save_images (bool, optional): Whether to save the output segmentations (as long format). Defaults to True.
            save_probs (bool, optional): Whether to save the probabilities (per channel). Defaults to False.
        """

        self.net = net
        self.dataset = dataset
        self.savepath = savepath
        self.params = params
        self.device = device
        self.savename = savename
        self.save_images = save_images
        self.save_probs = save_probs

        self.im_savefolder = os.path.join(
            self.savepath, savename + 'predictions')

        self.logger = logging.getLogger('__name__')

    def evaluate(self):
        """ Performs evaluation on given dataset
        """

        self.make_data_generator()
        self.metrics = []
        self.sample_names = []
        self.net.eval()

        self.logger.info(f'Starting to evaluate: {self.savename}')

        for (inputs, labels, sample_name) in self.datagenerator:

            with torch.no_grad():
                outputs = self.net(inputs.to(self.device))

            # apply softmax activation and get class predictions
            probs = torch.softmax(outputs.cpu(), dim=1)  # [B x C x H x W x D]
            predictions_long = torch.argmax(
                probs, dim=1)  # [B x H x W x D]

            # Convert to one-hot
            onehot_pred = torch.eye(outputs.shape[1])[
                predictions_long].permute(0, -1, 1, 2, 3).\
                contiguous()

            # remove bg channel
            probs = probs[:, 1:]
            predictions = onehot_pred[:, 1:]
            labels = labels[:, 1:]

            # compute dice values
            dice_list = calc_dice(predictions,
                                  labels, classsumm=False)  # [B, C - 1]

            # Replace Dice values where gt is only zeros to NaN
            for batchnum in range(dice_list.shape[0]):
                for channel in range(dice_list.shape[1]):
                    num_classes = len(torch.unique(labels[batchnum,
                                                          channel]))
                    if num_classes == 1:
                        dice_list[batchnum, channel] = float('nan')

            # Compute surface metrics (surface_eval = [B, C-1, 3])
            names, surface_eval = get_surface_metrics(
                predictions.numpy(), labels.numpy())

            # combine dice with surface metrics
            metric_names = ['Dice'] + names
            metrics = np.concatenate(
                (dice_list[:, :, np.newaxis].numpy(), surface_eval),
                axis=-1)  # [B, C-1, 4]

            # put metrics for batch in list with previous batches
            for k, elem in enumerate(metrics):
                self.metrics.append(elem)  # elem: [C-1, 4]
                self.sample_names.append(sample_name[k])

                # log the DSC for all structures
                dicescores = ["%.3f" % x for x in dice_list[k].tolist()]
                self.logger.info(f'{sample_name[k]} DSC: {dicescores}')

            # save resulting predictions (and probs) from batch
            if self.save_images:
                self.save_predictions(predictions_long, probs, sample_name)

        # Save metrics to files
        self.save_metrics(self.metrics, metric_names, self.sample_names)

    def save_metrics(self, metric_values, metric_names, sample_names,
                     addsavename=''):
        """  Writes all computed metrics to an excel file, one sheet per metric

        Args:
            metric_values (list[array]): list with for every sample a (num structures x num metrics) array
            metric_names (list): metric names
            sample_names (list): list of sample names
            addsavename (str, optional): Name added to filename of output xlsx. Defaults to ''.

        """

        if not os.path.isdir(self.savepath):
            os.makedirs(self.savepath)

        writer = pd.ExcelWriter(os.path.join(self.savepath,
                                             addsavename +
                                             self.savename +
                                             '_evalmetrics' +
                                             '.xlsx'), engine='openpyxl')

        for metric in range(len(metric_names)):
            metric_list = []
            for i, scan in enumerate(sample_names):
                scan_info = metric_values[i]  # [C-1, 4]
                metric_info = scan_info[:, metric].tolist()
                metric_list.append([scan] + metric_info)

            df = pd.DataFrame(metric_list, columns=[
                'Scan_Num', *self.params.structures])

            # set inf values to NaN (for Dice)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Add mean and std as rows
            df.loc['mean'] = df.mean()
            df.loc['std'] = df.std()

            df.to_excel(
                writer, sheet_name=metric_names[metric], float_format='%0.2f')

        writer.save()
        writer.close()

    def make_data_generator(self):
        """ Make datagenerators
        """
        # parameters
        loader_params = {
            'batch_size': self.params.val_batchsize, 'shuffle': False,
            'drop_last': False}

        # If evaluating test set use seperate class
        if 'test' in self.savename:
            data_set = MyManualTestset(
                self.dataset, self.params)
        else:
            data_set = MyDataset_Preloader(self.dataset, self.params)

        # Make dataloader
        self.datagenerator = torch.utils.data.DataLoader(
            data_set, **loader_params)

    def save_predictions(self, predictions, probs, sample_name):
        """ Saves segmentation predictions from model, predictions are saved in long format, 
        and probabilites are saved per class

        Args:
            predictions (tensor): Output predictions of network in long format [B, H, W, D]
            prob (tensor): Output probabilities of network in one-hot format [B, C, H, W, D]
            sample_name (str): sample file name
        """

        batch_size = probs.size(0)
        for j in range(batch_size):

            # save predictions as multi-class image
            multicl_predictions = predictions.cpu().numpy().astype(np.uint8)[j]

            saveas_MHA_fun(multicl_predictions,
                           os.path.join(self.im_savefolder,
                                        'multiclasspredictions_' +
                                        str(sample_name[j]) + '.mha'))

            # for probabilities, save per class
            if self.save_probs:
                for channel in range(len(self.params.structures)):
                    probs_im = probs[j, channel].cpu().numpy()

                    saveas_MHA_fun(probs_im.astype(np.float32), os.path.join(
                        self.im_savefolder, self.params.structures[channel] +
                        '_probs_' + str(sample_name[j]) + '.mha'))


def evaluator_runner(net, params, dataset, device, savepath,
                     save_images=False, save_probs=False):
    """ Runs the evaluator class for every datasplit

    Args:
        net (torch model): network with trained weights loaded in
        params (dataclass): contains all parameters
        dataset (dict): dataset containing keys:filenames for every datasplit
        device (torch device): where evaluation is performed
        savepath (str): where output is saved
        save_images (bool, optional): whether predicted segmentations are saved. Defaults to True.
        save_probs (bool, optional): whether predicted probabilites are saved. Defaults to False.
    """

    for key in dataset.keys():
        evaluator = Evaluator(net=net,
                              params=params,
                              dataset=dataset[key],
                              device=device,
                              savepath=savepath,
                              savename=key,
                              save_images=save_images,
                              save_probs=save_probs)
        evaluator.evaluate()
