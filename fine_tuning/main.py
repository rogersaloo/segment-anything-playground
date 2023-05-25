import numpy as np
import cv2
import torch
from statistics import mean
from tqdm import tqdm
from torch.nn.functional import threshold, normalize
from segment_anything import SamPredictor, sam_model_registry

import config
import plots
from preprocess_image import Mask
from benchmark_repo.segment_anything.utils.transforms import ResizeLongestSide
from variables import DataPath, GroundTruth


# set path variables to the masks, train and test data
ground_truth_masks = DataPath.ground_truth_masks
train_bottles = DataPath.train_bottles
test_bottles = DataPath.test_bottles
s_ground_truth_image = GroundTruth.single_ground_truth_image
preprocess_image = Mask(ground_truth_masks, train_bottles)


def compare_bbox_images_to_ground_truth_segmentation():
    """Get the bounding boxes and plot against masks ground truth

    Returns:
        image: passed image with bounding box and mask
    """
    bounding_box_coordinates = preprocess_image.get_bounding_box_coordinates()
    return plots.plot_ground_truth(test_bottles, s_ground_truth_image, bounding_box_coordinates)


def get_ground_truth_masks():
    """Obtain the masks of the ground truth bottles

    Returns:
        dict: dict of boolean ground truth masks
    """
    g_truth_masks = preprocess_image.get_ground_truth_segmentation_masks()
    return g_truth_masks


processed_gtruth_masks = get_ground_truth_masks()

# FINE-TUNING loading checkpoint of the SAM weights
sam_model = sam_model_registry[config.model_type](checkpoint=config.checkpoint)
sam_model.to(config.device)
sam_model.train()

# Preprocess image
transformed_data = preprocess_image.preprocess_image(sam_model)

# Setting Hyperparameters
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=config.lr, weight_decay=config.wd)
loss_fn = torch.nn.MSELoss()
bbox_coords = preprocess_image.get_bounding_box_coordinates()
keys = list(bbox_coords.keys())


# Training loop
def train_sam():
    """Train sam model on fine-tuned parameter

    Returns:
        dict: Returns a dict of mean losses
    """
    losses = []

    for epoch in range(config.num_epochs):
        epoch_losses = []
        # Train on the train images
        for k in tqdm(keys):
            input_image = transformed_data[k]['image'].to(config.device)
            input_size = transformed_data[k]['input_size']
            original_image_size = transformed_data[k]['original_image_size']

            # No grad here as we don't want to optimise the encoders
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image)

                prompt_box = bbox_coords[k]
                transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                box = transform.apply_boxes(prompt_box, original_image_size)
                box_torch = torch.as_tensor(box, dtype=torch.float, device=config.device)
                box_torch = box_torch[None, :]

                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )
            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(
                config.device)
            binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

            if isinstance(processed_gtruth_masks[k], bool):
                binary_mask_shape = upscaled_masks.shape[-2:]
                gt_binary_mask = torch.ones(binary_mask_shape, dtype=torch.float32).to(config.device) if \
                    processed_gtruth_masks[
                        k] else torch.zeros(binary_mask_shape, dtype=torch.float32).to(config.device)
            else:
                print(f"ERROR: ground_truth_masks[k] is not a boolean for key {k}")
                continue

            loss = loss_fn(binary_mask, gt_binary_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            losses.append(epoch_losses)
        print(f'EPOCH: {epoch}  == {mean(epoch_losses)}')

    return losses


def original_sam_model():
    """Download original SAM model and instantiate

    """
    sam_model_orig = sam_model_registry[config.model_type](checkpoint=config.checkpoint)
    sam_model_orig.to(config.device)
    return SamPredictor(sam_model_orig)


def tuned_sam_model():
    """Download tuned SAM model and instantiate

    """
    # Set up predictors for both tuned and original models
    return SamPredictor(sam_model)


def load_image_to_predict(image_id):
    """
    Args:
        image_id (str): id of the image in the dir to load

    Returns:
        image: loaded image for prediction
    """
    image = cv2.imread(f'{test_bottles}/{image_id}.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def predict_on_tuned_sam(k):
    """
    Args:
        k (str): image id in the dir to load

    Returns:
        model: tuned SAM model
    """
    # The model has not seen train images therefore you test the masking of the bottles
    image = load_image_to_predict(k)
    input_bbox = np.array(bbox_coords[k])
    predictor_tuned = tuned_sam_model()
    predictor_tuned.set_image(image)
    masks_tuned, _, _ = predictor_tuned.predict(
        point_coords=None,
        box=input_bbox,
        multimask_output=False, )
    return masks_tuned


def predict_on_original_sam(k=s_ground_truth_image):
    """
    Args:
        k (str): image id in the dir to load

    Returns:
        model: original SAM model
    """
    image = load_image_to_predict(k)
    input_bbox = np.array(bbox_coords[k])
    predictor_original = original_sam_model()
    predictor_original.set_image(image)

    masks_orig, _, _ = predictor_original.predict(
        point_coords=None,
        box=input_bbox,
        multimask_output=False, )
    return masks_orig


def main():
    """Run the training model
    """
    compare_bbox_images_to_ground_truth_segmentation()  # compare bbox vs mask compatibility
    losses = train_sam()  # Train tuned model & get loss
    plot_training_loss = plots.plot_train_mean(losses)  # plot loss graph

    # # Plot mask comparison between tuned and original model
    # image = load_image_to_predict(s_ground_truth_image)  # load image to predict
    # input_bbox = np.array(bbox_coords[s_ground_truth_image])  # bounding box for prompting sam
    # masks_orig = predict_on_original_sam(s_ground_truth_image)  # mask image on original model
    # mask_tuned = predict_on_tuned_sam(s_ground_truth_image)  # mask image on tuned model
    # plots.compare_models_masks(image, input_bbox, mask_tuned, masks_orig)  # plot masks for both models


if __name__ == '__main__':
    main()
