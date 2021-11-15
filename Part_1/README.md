# PART 1
_______________________________________________________________________

## INTRODUCTION 

This is a simple code which uses an in-built tensorflow datasets to build MNIST classifier. It also allows the use of a custom dataset to create a simple image classifier

## How to use it 

1. Set up environment by installing the packages in requirements.txt
```bash
pip3 install -r requirements.txt
```

#### Using MNIST dataset
2. Go to the root directory where train.py is then run the following commands.
```bash
cd Part_1
python train.py
```

#### Using custom dataset
To use a custom dataset you have to pass these arguments to the command line:
> --train_dataset_path "path to directory containing train images in their class folders"
> 
> --val_dataset_path "Path to directory containing the different classes for the val data"

3. To get all the possible argument that can be passed run the following command:
```bash
python train.py -h  
```

4. Quick example of building with custom dataset
```bash
cd Part_1
python train.py --train_dataset_path ./train --val_dataset_path ./val --target_size 128 --num_classes 3 --epoch 2  
```
