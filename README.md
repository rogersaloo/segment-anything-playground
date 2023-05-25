## Description
SAM model on the root of this repository ```official_sam_repo``` provides a foundational model for segmentation tasks through prompting.
The model release occurred without an explicit fine-tuning functionality. This folder contains a finetune of the model.

The fine-tuning here focuses on the mask decorder only because it is lighter and therefore faster an more effecient to 
fine tune.

You do not use the predictor SamPredctor in ```segment_anything/predictor.py``` since the predictor contains all the three
parts of the underlying architecture (image encoder, prompt encoder and mask decoder). Additionally, the "predict_torch" 
in the SamPredictor class has a decorator(@torch.no_grad) preventing the re-computation of gradients.

## Fine-tuning
In normal case of computer vision if we try to show a model new images that are not the same as the training dataset, the 
output performance will be degraded. Fine-tuning helps to maintain good performace comparable to the original model.

Fine-tuning allows uptake of the pre-trained model and its weights and subjecting it to a new dataset or a task specific 
task relating to the needs of a specific use case.

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
You download the base model checkpoint and load it. We make sure to freeze the large  encoder

### Dataset
I use the MVTec AD dataset specifically the
 bottles dataset from [kaggle](https://www.kaggle.com/datasets/ipythonx/mvtec-adsince)
I feel that the data is unlikely to have seen by SAM. Even if the data was seen the aim of the repo is just to showcase fine-tuning for SAM. 
The dataset has precise ground truth segmentation masks and create bounding boxes on them which can be used as prompts to SAM for segmentation.

### Finetuning
SAM model realease occured without any explicit fine-tuning functionality. In this notebook you will fine-tune the image decorder only by;

#### Tasks

1. Wrap image encoder with no gradient flow
2. Generate prompt embeds with no grad flow
3. Generate the masks -(use case output mask 1)

#### Hyper-params

1. Learning rate to 1e-06 
2. Change image decoder loss type to MSE
3. Using the Adam optimizer on the image decoder

## Result
The masking results from the new model are not perfect, but showcase an improvement from the base SAM model on this task.
![fine_tune_mask.png](assets%2Ffine_tune_mask.png)

<a target="_blank" href="https://colab.research.google.com/drive/144kBr52E_X3hElaJw65aDgKtueKtXOCx">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### See documentation
* `mkdocs build` - Build the documentation site.
* `mkdocs serve` - Start the live-reloading docs server.

### Resources 
1. Original SAM paper
2. segment-anything github repo
3. Some code implementation from encords open-source framework for computer vision [here](https://github.com/encord-team) by [Alex Bonnet](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/) .
4. MTVEC AD Data set
Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,
"A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection",
IEEE Conference on Computer Vision and Pattern Recognition, 2019
