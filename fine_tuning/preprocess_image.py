import cv2
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from fine_tuning.benchmark_repo.segment_anything.utils.transforms import ResizeLongestSide
from fine_tuning.config import model_type, checkpoint, device, lr, wd, num_epochs


def bbox_coords(truth_masks) -> dict:
    """
  Returns:
      dict: coordinates of bounding boxes of the passed image
  """
    bbox_coordinates = {}
    for f in sorted(Path(f'{truth_masks}').iterdir())[:100]:
        image_name = f.stem[:-5]  # stem the images name
        image = cv2.imread(f.as_posix())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        x, y, w, h = cv2.boundingRect(contours[0])
        height, width, _ = image.shape
        bbox_coordinates[image_name] = np.array([x, y, x + w, y + h])
    return bbox_coordinates


def ground_truth_masks(train_bottles: str,
                       bbox_coordinates: dict) -> dict:
    """Take a look at the images, the bounding box prompts and the ground truth segmentation masks

  Returns:
      dict: dict extract ofthe ground truth segmentation masks
  """
    ground_truth_masks = {}
    for k in bbox_coordinates.keys():
        gt_grayscale = cv2.imread(f'{train_bottles}/{k}-mask.png', cv2.IMREAD_GRAYSCALE)
        ground_truth_masks[k] = (gt_grayscale == 0)
    return ground_truth_masks


def preprocess_image(train_bottles, model) -> dict:
    """Returns pre-processed images
    Image resizing by convert the input images into a format SAM's internal functions expect"""
    transformed_data = defaultdict(dict)
    for k in bbox_coords.keys():
        image = cv2.imread(f'{train_bottles}/{k}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = ResizeLongestSide(model.image_encoder.img_size)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=device)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        input_image = model.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])

        transformed_data[k]['image'] = input_image
        transformed_data[k]['input_size'] = input_size
        transformed_data[k]['original_image_size'] = original_image_size
    return transformed_data


def main():
    boxes = bbox_coords()
    print(boxes)


if __name__ == '__main__':
    main()