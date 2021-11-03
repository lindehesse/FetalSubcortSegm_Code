# Subcortical Segmentation of the Fetal Brain from 3D Ultrasound

This code contains the PyTorch training code for performing multi-label segmentation with a 3D U-Net. It was written to perform subcortical segmentation in the fetal brain from 3D Ultrasound but can be used for other multi-label segmentation tasks.
## Usage

1. Install all packages from requirements.txt
2. Adapt 'params.json' or 'define_parameters.py' for training (see parameter files below)
3. Call code with:
        
        python train_runner.py -p parameterpath -d datapath -s savepath
        e.g.: python train_runner.py -p "config/params.json" -d 'exampledata' -s 'savefolder'


The code can also be run to just evaluate a previously trained model, the parameter 'mode' needs then to be changed to 'eval_only' and the code can be run with:

        python train_runner.py -p parameterpath -d datapath -s savepath -e evalfolder

evalfolder needs to be set to the folder containing 1 or multiple training runs. It searches for 'best_model_los.tar' in the respective folder, and can thus also evaluated multiple nested runs. 

## Parameters Files
All parameters are defined in 'define_parameters.py' in a Dataclass, with default values. Changes to parameters can be either done in that file (changing defaults) or in 'config/params.json' (will replace default values during training).

## Dependencies
```
python==3.7
torch==1.7.1
torchvision==0.8.2
pandas==1.1.3
scipy==1.6.1
simpleitk==2.0.2
torchviz==0.0.1
tensorboard==2.4.1
openpyxl==3.0.5
xlrd==1.2.0
scikit-image == 0.18.1
dataclass-json == 0.5.3
enforce-typing == 1.0.0.
torchio == 0.18

surface_distance:
# git clone https://github.com/lindehesse/surface-distance.git
# pip install surface-distance/

```

## File Structure
An example of the expected file structure is given in the folder exampledata. The folder contains dummy data to verify all code is running correctly, but does not contain actual image volumes. 

Expected File Structure (all names can be set in define_parameters.save_params):
```
traindata  
│
└───datasplit_foldername
│  		  datasplitfilename
│   
└───dataset_train
│	└───volume_names
│	│		vol1.mha
│	│		vol2.mha
│	│		...
│	└───structure1
│	│		vol1.mha
│	│		vol2.mha
│	│		...
│	└───structure2
│	│		vol1.mha
│	│		vol2.mha
│	│		...
│	└─── ...
│
└───dataset_test
	└───(same structure as dataset_train)

```

The data can also be in nested folder as the code searches for the correct file in 'dataset_train' based on the filename in the datasplit file. 

Volume extension (also defined in parameters) can be everything that can be read by SimpleITK. 

Structure names (structure 1, structure 2, ...) can be set in parameter settings and refer to the name of the folder of the respective structure. Each masks (having the same name as the image volume) contains the binary ground-truth mask for a single structure. 

## Datasplit
An example datasplit is given in the folder 'exampledata/datasplits' . The datasplit file should be a json file with keys for: 'test', 'train', 'validation'. Code searches for train/validation files in 'dataset_train', and for the test files in 'dataset_test'. 

