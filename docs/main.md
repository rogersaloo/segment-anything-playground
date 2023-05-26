# Train
## Description
Contains the main script that is used to run and fine tune the SAM model.

## Modules
1. ```compare_bbox_images_to_ground_truth_segmentation```: Get the bounding boxes and plot against masks ground truth.
2. ```get_ground_truth_masks```: Obtain the masks of the ground truth bottles
3. ```train_sam```: Train sam model on fine-tuned parameter
4. ```original_sam_model```: Download original SAM model and instantiate
5. ```tuned_sam_model```: Download tuned SAM model and instantiate
6. ```load_image_to_predict```: loaded image for prediction
7. ```predict_on_tuned_sam```: tuned SAM model
8. ```predict_on_original_sam```: image id in the dir to load
9. ```main()```: Run the training model