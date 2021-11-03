from dataclasses import dataclass, field
import json
from dataclasses_json import dataclass_json
from typing import List, Type
from enforce_typing import enforce_types
from datetime import datetime
import os
from pathlib import Path


@enforce_types
@dataclass(frozen=True)
class AugmParams:
    hor_flips: bool = True
    scaling_factor: float = 1.5  # between 1/1.5 and 1.5 (set to 1 to disable)
    rotation_degrees: int = 20  # set to 0 to disable
    trans_pixels: int = 10  # set to 0 to disable


@enforce_types
@dataclass(frozen=True)
class NetwParams:
    min_featuremaps: int = 16 # no. featuremaps in first layer
    depth: int = 5  # depth of the U-Net


@enforce_types
@dataclass(frozen=True)
class SaveParams:
    save_predictions: bool = True  # whether to save binary predictions
    save_probs: bool = False   # whether predicted probs are saved
    checkpoint_frequency: int = 20  # no. epochs after which checkpoint are saved
    validation_frequency: int = 1  # no. epochs where validation is performed
    num_lastmodels: int = 5  # no. of models which are saved at the end
    
    # whether to save network architecute as im (set to false if getting graphviz errors)
    architecture_saveasim: bool = False


@enforce_types
@dataclass(frozen=True)
class DataParams:
    # File structure (for more info see readme)
    datasplit_foldername: str = "datasplits"
    datasplit_filename: str = "datasplit.json"
    dataset_train: str = "traindata"
    dataset_test: str = "testdata"
    volume_names: str = "images" 
    extension: str = '.mha' # extension of datavolumes


@enforce_types
@dataclass_json
@dataclass
class Parameters:

    # Data and Save paths
    exp_name: str  # will be inferred from parametername
    datapath: str = 'exampledata' 
    savepath: str = 'savefolder'

    mode: str = 'train'  # can be 'eval_only' or 'train'
    assert mode in ['train', 'eval_only'], f"Given mode is not implemented: {mode}"

    n_epochs: int = 2  # no. training epochs
    preload_data: bool = False  # whether to load all data in gpu memory
    num_runs: int = 1  # number of training runs (to account for variability)

    # Optimzer and loss parameters
    lr: float = 0.001
    optimizer: str = 'Adam'  # currently implemented: 'Adam' or 'SGD'
    loss: str = "DiceAndCE"  # currently implemented: 'DiceAndCE'
    CE_weight: float = 1.0  # loss = dice + self.CE_weight * CE
    eval_loss: str = loss  # Loss which is used for validation set

    # Batchsizes
    train_batchsize: int = 1
    val_batchsize: int = 1

    # Other settings
    augm_params: AugmParams = AugmParams()
    netw_params: NetwParams = NetwParams()
    save_params: SaveParams = SaveParams()
    data_params: DataParams = DataParams()

    # Structures  and weeks used during training
    structures: List[str] = field(default_factory=lambda: [
        "CP",
        "LPVH",
        "CSPV",
        "CB"
    ])


    # Only used for mode = eval_only
    evalpath: str = ''  # folder where trained model is located

    def __post_init__(self):
        # Set actual data and save paths by combining main paths with datanames
        self.testset_path: str = os.path.join(self.datapath,
                                              self.data_params.dataset_test)
        self.trainset_path: str = os.path.join(self.datapath,
                                               self.data_params.dataset_train)
        self.datasplit_path: str = os.path.join(self.datapath, 
                                                self.data_params.datasplit_foldername)
        
        # Get the name from the .json file, but without extension
        date = datetime.now().strftime('%d_%m_%Y_')
        self.savepath_exp: str = os.path.join(
            self.savepath, date + Path(self.exp_name).stem)

        # Assert that an eval path is given when evaluating only
        if self.mode == 'eval_only':
            assert self.evalpath != '' , "No evaluation folder given"