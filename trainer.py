import json
import pandas as pd
import os
import logging
import time
from torch import optim
import torch
import glob


import utils as utils
from lossfile import DiceAndCE
from datagenerator import MyDataset, MyDataset_Preloader
from pathlib import Path


class Trainer:
    """
    Class to train the model
    """

    def __init__(self,
                 network,
                 dataset,
                 device,
                 savefolder,
                 params):
        """Class to train the model

        Args:
            network (network class): network to be trained
            dataset (dict): contains keys for 'test', 'train'
            and 'train_selection'
            device (device): pytorch device
            savefolder (str): folder to save everything to
            params (dataclass): contains all parameters
        """
        self.logger = logging.getLogger('__name__')
        self.logger.info('initialize trainer')

        # Set function arguments
        self.dataset = dataset
        self.params = params
        self.device = device
        self.net = network
        self.savefolder = savefolder

        # Initialize variables
        self.optimizer = None
        self.criterion = None
        self.eval_criterion = None
        self.val_loss = []
        self.saveloss = []

        # Prepare for training
        self.set_optimizer()
        self.set_loss()

    def train_model(self):
        """ Perform training loop
        """

        self.make_data_generators()
        self.save_parameters()

        n_epochs = self.params.n_epochs
        best_loss = None

        self.logger.info('Start training model')
        for epoch in range(n_epochs):
            epoch_loss = 0
            time_now = time.time()
            self.net.train()
            for (input, labels, filenames) in self.training_generator:

                self.optimizer.zero_grad()
                outputs = self.net(input.to(self.device))
                loss = self.criterion(outputs, labels.to(self.device))

                del outputs, input, labels, filenames

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss

            training_time = time.time() - time_now

            # Perform validation on validation set (called train_selection)
            if epoch % self.params.save_params.validation_frequency == 0 or \
                    (epoch in range(n_epochs - self.params.save_params.num_lastmodels,
                                    n_epochs)):
                best_loss = self.run_validation(epoch, epoch_loss.item(),
                                                best_loss)

            # log all info from epoch
            item_loss = epoch_loss.item()/len(self.training_generator)
            self.logger.info(f"[{epoch}] training loss: {item_loss:.6f}")
            self.logger.info(
                f"[{epoch}] traintime [min]: {training_time/60:.3f}")
            self.logger.info(
                f"[{epoch}] traintime incl val [min]: {(time.time() - time_now)/60:.3f}")

    def make_data_generators(self):
        """ Initializes data generators
        """
        self.logger.info('making data generators')

        # data transforms
        params_train = {
            'batch_size': self.params.train_batchsize, 'shuffle': True,
            'num_workers': 8, 'drop_last': True}
        params_val = {
            'batch_size': self.params.val_batchsize, 'shuffle': False,
            'num_workers': 8, 'drop_last': False}

        # make datasets
        if self.params.preload_data:
            training_set = MyDataset_Preloader(
                self.dataset['train'], self.params,
                **vars(self.params.augm_params))
            val_set = MyDataset_Preloader(
                self.dataset['validation'], self.params)
        else:
            training_set = MyDataset(
                self.dataset['train'], self.params,
                **vars(self.params.augm_params))
            val_set = MyDataset(
                self.dataset['validation'], self.params)

        # make datagenerators
        self.training_generator = torch.utils.data.DataLoader(
            training_set, **params_train)
        self.validation_generator = torch.utils.data.DataLoader(
            val_set, **params_val)

        # Save used datasplit
        self.save_datasplit()

    def set_optimizer(self):
        """ Set the optimizer used in training

        Raises:
            NotImplementedError: raised when defined optimiser is not known
        """
        optimizer = self.params.optimizer
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.net.parameters(), lr=self.params.lr)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.net.parameters(), lr=self.params.lr)
        else:
            raise NotImplementedError('Defined optimizer is not implemented')

    def choose_loss(self, loss_type):
        """helper function to set the losses for evaluation and training

        Args:
            loss_type (str): name of loss
            criterion (loss object): loss criterion to which this loss is assigned

        Raises:
            NotImplementedError: raised when defined loss is not known
        """
        if loss_type == 'DiceAndCE':
            criterion = DiceAndCE(
                CE_weight=self.params.CE_weight)
        else:
            raise NotImplementedError('Defined loss is not implemented')
        return criterion

    def set_loss(self):
        """Set loss for training and evaluation
        """

        self.criterion = self.choose_loss(self.params.loss)
        self.eval_criterion = self.choose_loss(self.params.eval_loss)

    def run_validation(self, epoch, epoch_loss, best_loss):
        """ Runs and saves all validation information

        Args:
            epoch (int): number of epoch
            epoch_loss (float): loss at current epoch
            best_loss (float): best loss

        Returns:
            float: best loss
        """
        # Run validation
        val_loss = self.eval_validation()       

        # Logs and saves everything
        self.val_loss.append(val_loss.item())

        # Running average over last 5 epochs
        if len(self.val_loss) > 5:
            self.val_loss.pop(0)
        running_average = sum(self.val_loss) / len(self.val_loss)
        self.logger.info(
            f"[{epoch}] running validation loss [5 epochs]: {running_average:.6f}")

        # save as best model if running_average is better
        if best_loss is None or running_average < best_loss:
            torch.save({'epoch': epoch,
                        'model_state_dict': self.net.state_dict(),
                        'optimizer_state_dict':
                        self.optimizer.state_dict(),
                        'loss': self.saveloss}, self.best_model_path())
            best_loss = running_average

            # save current epoch to txt file
            textname = os.path.join(os.path.dirname(self.best_model_path()),
                                    'best model_epoch.txt')
            with open(textname, 'w') as file:
                file.write('Best Model Has been saved on epoch: ' + str(epoch))

        self.saveloss.append(
            [epoch, epoch_loss / len(self.training_generator),
             val_loss.item(), running_average])

        # write losses to csv
        df = pd.DataFrame(self.saveloss, columns=[
            'epoch', 'train_loss', 'val_loss', 'running val [5 epochs]'])
        df.to_csv(os.path.join(self.savefolder, 'losses.csv'))

        # save model after this epoch (for cp frequency and last n models)
        last_epoch = self.params.n_epochs
        if epoch % self.params.save_params.checkpoint_frequency == 0 or \
            epoch in range(last_epoch - self.params.save_params.num_lastmodels,
                           last_epoch):
            savename = 'modelcheckpoint_epoch_%d_loss%.6f.tar' % \
                (epoch, val_loss.item())
            torch.save({'epoch': epoch,
                        'model_state_dict': self.net.state_dict(),
                        'optimizer_state_dict':
                        self.optimizer.state_dict(),
                        'loss': self.saveloss},
                       os.path.join(
                self.savefolder, savename))

        return best_loss

    def eval_validation(self):
        """ Performs actual validation

        Returns:
            float: validation loss
        """
        # creates batch of validation data
        running_val_loss = 0
        total_volumes = 0

        for (input, labels, filenames) in self.validation_generator:
            with torch.no_grad():
                outputs = self.net(input.to(self.device))

            loss = self.eval_criterion(outputs, labels.to(self.device))

            # Account for last batch which might be smaller
            total_volumes = total_volumes + input.size(0)
            running_val_loss += loss * input.size(0)

            del outputs, input, labels, filenames

        val_loss = running_val_loss / total_volumes
        return val_loss

    def best_model_path(self):
        """ Returns path of best model (based on val loss)

        Returns:
            str: path
        """
        return os.path.join(self.savefolder, 'best_model' + '_loss.tar')

    def last_checkpoint_path(self):
        """ Returns path of last saved model

        Returns:
            str: path
        """
        last_epochmodel = glob.glob(os.path.join(self.savefolder,
                                                 '*epoch*' +
                                                 str(self.params.n_epochs - 1)
                                                 + '*.tar'))
        return last_epochmodel[0]

    def save_datasplit(self):
        """  Saved the filenames used for training (datasplit)
        """
        # Convert dataset to filenames instead of full paths
        file_dict = {}
        for key in self.dataset.keys():
            pathnames = self.dataset[key]
            file_dict[key] = [Path(x).stem for x in pathnames]

        # Save datasplit
        with open(os.path.join(self.savefolder, 'datasplit.json'), 'w') as f:
            json.dump(file_dict, f, ensure_ascii=True, indent=4)

    def save_parameters(self):
        """ Save the used parameters as json
        """
        data_dict = self.params.to_dict()
        utils.save_json(data_dict, os.path.join(self.savefolder,
                                                'config.json'))


