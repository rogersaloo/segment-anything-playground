import numpy as np
import cv2
import torch
from statistics import mean
from torch.nn.functional import threshold, normalize
from segment_anything import SamPredictor, sam_model_registry
from variables import DataPath, GroundTruth
import config
from preprocess_image import Mask
from utils.plots import Plots

# set path variables to the masks, train and test data
ground_truth_masks = DataPath.ground_truth_masks
train_bottles = DataPath.train_bottles
test_bottles = DataPath.test_bottles
ground_truth_image = GroundTruth.ground_truth_image

preprocess_image = Mask(ground_truth_masks,train_bottles)
def compare_bbox_images_to_ground_truth_segmentation():
    bbox_coordinates = preprocess_image.bbox_coords()
    print(bbox_coordinates)
    # g_truth_masks = preprocess_image.ground_truth_masks(train_bottles, bbox_coordinates)
    # return Plots.plot_ground_truth(test_bottles, ground_truth_image, bbox_coordinates)


#FINETUNING
# loading checkpoint of the SAM weights
sam_model = sam_model_registry[config.model_type](checkpoint=config.checkpoint)
sam_model.to(config.device)
sam_model.train()

# Preprocess image
transformed_data = preprocess_image.preprocess_image(sam_model)

# Setting Hyperparameters
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=config.lr, weight_decay=config.wd)
loss_fn = torch.nn.MSELoss()
bbox_coords = preprocess_image.bbox_coords()
keys = list(bbox_coords.keys())

# Training loop
def train_sam(transform=None):
    losses = []

    for epoch in range(config.num_epochs):
        epoch_losses = []
        # Train on the train images
        for k in keys:
            input_image = transformed_data[k]['image'].to(config.device)
            input_size = transformed_data[k]['input_size']
            original_image_size = transformed_data[k]['original_image_size']

            # No grad here as we don't want to optimise the encoders
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image)

                prompt_box = bbox_coords[k]
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

            upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(config.device)
            binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)
                                    )
            if isinstance(ground_truth_masks[k], bool):
                binary_mask_shape = upscaled_masks.shape[-2:]
                gt_binary_mask = torch.ones(binary_mask_shape, dtype=torch.float32).to(config.device) if ground_truth_masks[
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
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')
        return losses


def original_sam_model():
    # Load up the model with default weights
    sam_model_orig = sam_model_registry[config.model_type](checkpoint=config.checkpoint)
    sam_model_orig.to(config.device)
    return SamPredictor(sam_model_orig)


def tuned_sam_model():
    # Set up predictors for both tuned and original models
    return SamPredictor(sam_model)

def load_image_to_predict(k):
    image = cv2.imread(f'{test_bottles}/{k}.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def predict_on_tuned_sam(k=5):
    # The model has not seen train images therefore you test the masking of the bottles
    image = load_image_to_predict(k)
    input_bbox = np.array(bbox_coords[k])
    predictor_tuned = tuned_sam_model()
    predictor_tuned.set_image(image)
    masks_tuned, _, _ = predictor_tuned.predict(
        point_coords=None,
        box=input_bbox,
        multimask_output=False, )

def predict_on_tuned_sam(k=5):
    image = load_image_to_predict(k)
    input_bbox = np.array(bbox_coords[k])
    predictor_original = original_sam_model()
    predictor_original.set_image(image)

    masks_orig, _, _ = predictor_original.predict(
        point_coords=None,
        box=input_bbox,
        multimask_output=False,)


def main():
    compare_bbox_images_to_ground_truth_segmentation()

if __name__ == '__main__':
    main()

