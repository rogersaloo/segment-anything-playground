# Preprocesing_Images

Preprocessing images before loading to the SAM model. The modules resize the image size to fit the input of a SAM model.

## modules
1. ```get_bounding_box_coordinates```: coordinates of bounding boxes of the passed image
2. ```preprocess_image```: Extract the ground truth segmentation masks
3. ```preprocess_image```: Returns pre-processed images Image resizing by convert the input images into a format SAM's internal functions expect