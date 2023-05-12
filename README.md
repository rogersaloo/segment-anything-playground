# Fine-tuning the SAM model on custom images
## Fine-tuning
In normal case of computer vision if we try to show a model new images that are not the same as the training dataset, the 
output performance will be degraded. Fine-tuning helps to maintain good performace comparable to the original model.

Fine-tuning allows uptake of the pre-trained model and its weights and subjecting it to a new dataset or a task specific 
task relating to the needs of a specific use case.


## Description
SAM model on the root of this repository provides a foundational model for segmentation tasks through prompting.
The model release occurred without an explicit fine-tuning functionality. This folder contains a finetune of the model.

The fine-tuning here focuses on the mask decorder only because it is lighter and therefore faster an more effecient to 
fine tune.

You do not use the predictor SamPredctor in ```segment_anything/predictor.py``` since the predictor contains all the three
parts of the underlying architecture (image encoder, prompt encoder and mask decoder). Additionally, the "predict_torch" 
in the SamPredictor class has a decorator(@torch.no_grad) preventing the re-computation of gradients.

## Requirements
To finetune the model you need the images to segment, ground truths to the image and prompts for the image. This example
 

## Dataset 
The dataset in this case is bottle defect data set which is not among the 23 provided dataset and may also not have been
for training SAM. Running inference with the pretrained weights performs well but not perfectly as compared to other 

## Get started
### Preprocess data
The image scans in RGB format are converted from numpy arrays to tensors and resize the images to tha shape of the size 
the transformer input inside the predictor.

## Setup
You download the base model checkpoint and load it. The set an Adam optimizer for the mask decoder only using an MSE loss.

## Training 

## Resources 
1. MTVEC AD Data set
Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,
"A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection",
IEEE Conference on Computer Vision and Pattern Recognition, 2019