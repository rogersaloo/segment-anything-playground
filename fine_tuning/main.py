import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from pathlib import Path
from statistics import mean
from collections import defaultdict
from torch.nn.functional import threshold, normalize
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamPredictor, sam_model_registry
from variables import DataPath, GroundTruth
from fine_tuning.config import model_type, checkpoint, device, lr, wd, num_epochs



#set path variables to the masks, train and test data
ground_truth_masks = DataPath.ground_truth_masks
train_bottles = DataPath.train_bottles
test_bottles = DataPath.test_bottles

# ground truth
ground_truth_image = GroundTruth.ground_truth_image

def bbox_coords() -> dict:
    """Returns dict array of bounding boxes for the """
    bbox_coords = {}
    for f in sorted(Path(f'{ground_truth_masks}').iterdir())[:100]:
      k = f.stem[:-5] #stem the images name
      im = cv2.imread(f.as_posix())
      gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
      contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
      x,y,w,h = cv2.boundingRect(contours[0])
      height, width, _ = im.shape
      bbox_coords[k] = np.array([x, y, x + w, y + h])
    return bbox_coords

def ground_truth_masks() -> dict:
    """Returns dict extract ofthe ground truth segmentation masks
    Take a look at the images, the bounding box prompts and the ground truth segmentation masks"""
    ground_truth_masks = {}
    for k in bbox_coords.keys():
      gt_grayscale = cv2.imread(f'{train_bottles}/{k}-mask.png', cv2.IMREAD_GRAYSCALE)
      ground_truth_masks[k] = (gt_grayscale == 0)
    return ground_truth_masks



# Helper functions provided in https://github.com/facebookresearch/segment-anything/blob/9e8f1309c94f1128a6e5c047a10fdcb02fc8d651/notebooks/predictor_example.ipynb
def show_mask(mask, ax, random_color=False):
    """Helper function for mask image"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    """Helper function for bounding box image"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))



def plot_ground_truth():
    """ Returns bounding box images
    You can see here that the ground truth mask is extremely tight which will be good for calculating an accurate loss.
    The bounding box overlays very well on the broken part.
    The bounding box overlaid will be a good prompt.
    """
    image = cv2.imread(f'{test_bottles}/{ground_truth_image}.png')
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_box(bbox_coords[ground_truth_image], plt.gca())
    plt.axis('off')
    plt.show()


#loading checkpoint of the SAM weights
sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)
sam_model.train()


def preprocess_image() -> None:
    """Returns pre-processed images
    Image resizing by convert the input images into a format SAM's internal functions expect"""
    transformed_data = defaultdict(dict)
    for k in bbox_coords.keys():
      image = cv2.imread(f'{train_bottles}/{k}.png')
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      transform = ResizeLongestSide(sam_model.image_encoder.img_size)
      input_image = transform.apply_image(image)
      input_image_torch = torch.as_tensor(input_image, device=device)
      transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

      input_image = sam_model.preprocess(transformed_image)
      original_image_size = image.shape[:2]
      input_size = tuple(transformed_image.shape[-2:])

      transformed_data[k]['image'] = input_image
      transformed_data[k]['input_size'] = input_size
      transformed_data[k]['original_image_size'] = original_image_size


# Set up the optimizer, learning rate and weight decay
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)
loss_fn = torch.nn.MSELoss()
keys = list(bbox_coords.keys())



# Training loop
losses = []

for epoch in range(num_epochs):
  epoch_losses = []
  # Train on the train images
  for k in keys:
    input_image = transformed_data[k]['image'].to(device)
    input_size = transformed_data[k]['input_size']
    original_image_size = transformed_data[k]['original_image_size']
    
    # No grad here as we don't want to optimise the encoders
    with torch.no_grad():
      image_embedding = sam_model.image_encoder(input_image)
      
      prompt_box = bbox_coords[k]
      box = transform.apply_boxes(prompt_box, original_image_size)
      box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
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

    upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
    binary_mask = normalize(threshold(upscaled_masks, 0.0, 0)
    )
    if isinstance(ground_truth_masks[k], bool):
        binary_mask_shape = upscaled_masks.shape[-2:]
        gt_binary_mask = torch.ones(binary_mask_shape, dtype=torch.float32).to(device) if ground_truth_masks[k] else torch.zeros(binary_mask_shape, dtype=torch.float32).to(device)
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

"""Loss misbehaving but can be fixed by tuning the hyper-parameters further. You can play with different learning rates."""

mean_losses = [mean(x) for x in losses]
mean_losses

plt.plot(list(range(len(mean_losses))), mean_losses)
plt.title('Mean epoch loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')

plt.show()

"""## Prediction Comparison
 We can compare our tuned model to the original model
"""

# Load up the model with default weights
sam_model_orig = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model_orig.to(device);

# Set up predictors for both tuned and original models
predictor_tuned = SamPredictor(sam_model)
predictor_original = SamPredictor(sam_model_orig)

# The model has not seen train images therefore you test the masking of the bottles
image = cv2.imread(f'{test_bottles}/{k}.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor_tuned.set_image(image)
predictor_original.set_image(image)

input_bbox = np.array(bbox_coords[k])

masks_tuned, _, _ = predictor_tuned.predict(
    point_coords=None,
    box=input_bbox,
    multimask_output=False,
)

masks_orig, _, _ = predictor_original.predict(
    point_coords=None,
    box=input_bbox,
    multimask_output=False,
)

"""We can see here that the tuned model is starting to mask a larger potion of the bottle than the original model. With further training, more data and further hyperparameter tuning we will be able to improve this result.


"""

def plot_sam_vs_tuned():
    """Returns an image """
    _, axs = plt.subplots(1, 2, figsize=(25, 25))


    axs[0].imshow(image)
    show_mask(masks_tuned, axs[0])
    show_box(input_bbox, axs[0])
    axs[0].set_title('Mask with Tuned Model', fontsize=26)
    axs[0].axis('off')


    axs[1].imshow(image)
    show_mask(masks_orig, axs[1])
    show_box(input_bbox, axs[1])
    axs[1].set_title('Mask with Untuned Model', fontsize=26)
    axs[1].axis('off')

    plt.show()