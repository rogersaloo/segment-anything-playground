# Fine-Tuning SAM Documentation
The documentation contains an explanation of the codes fine tuning process and preperation of scripts for production. Prepared by rogers aloo [github](https://github.com/rogersaloo) check the full repo to finetuning the SAM model [here](https://github.com/rogersaloo/segment-anything-playground).

## Description
SAM model on the root of this [repository](https://github.com/rogersaloo/segment-anything-playground) ```official_sam_repo``` provides a foundational model for segmentation tasks through prompting.
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

## Commands
To start and load the live server run;
* `mkdocs build` - Build the documentation site.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        <!-- index.md  # The documentation homepage. -->
        helpers.md  # helper function for plotting masks
        plots.md  # plots 
        preprocess_image.md # preprocessing images for use in sam
        main.md # main train and sam model
        config.md # fine tuning hyperparameters and variables
        app.md # dockerize and make fast api endpoint
        japanese.md # Japanese translation of the documentation
