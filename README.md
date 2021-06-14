# Learning Human Optical Flow

## General Info

Group: BananaPyjama

Members: Baptiste Freydt (bfreydt), Clement Guerner (cguerner), Luke Smith (lusmith)

Base Project: https://gitlab.ethz.ch/yanzhang25/mp21-proj6-humanof

## Description

This repo contains an exploration into the Machine Perception task of 'Learning Human Optical Flow' in which the velocity of a humans body parts in a sequence of images is deduced and indicated through colouring of the images.

For more details please refer to ```report.pdf```

## Reproducing results

Running the following commands require python 3.8 (only version we tested), and a venv with the packages listed in ```requirements.txt```. To install them on leonhard you can use 

```
$ source scripts/init_leonhard.sh
```

Then please specify the paths in each script you want to run.

To get the predictions on the test set from the pretrained model (with the venv activated):
```
$ ./scripts/test.sh
```

To get the predictions and EPE on the validation set from the pretrained model* (with the venv activated):
```
$ ./scripts/val.sh
```

To train (or at least fine tune) the model, we used the command:
```
$ ./scripts/train.sh
```

*: please note that for this final submission the model has been fine tuned on the validation set