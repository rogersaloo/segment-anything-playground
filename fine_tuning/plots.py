from statistics import mean
import cv2
import matplotlib.pyplot as plt

from helpers import show_box, show_mask

def plot_ground_truth(
        test_bottles: str,
        ground_truth_image: str,
        bbox_coords: dict):
    """plot ground truth image only

    Args:
        test_bottles (str): cracked bottle sample used for testing model
        ground_truth_image (str): ground truth image 
        bbox_coords (dict): bounding box coordinate from the images
    """
    image = cv2.imread(f'{test_bottles}/{ground_truth_image}.png')
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_box(bbox_coords[ground_truth_image], plt.gca())
    plt.axis('off')
    plt.show()


def plot_sam_vs_tuned(image, input_bbox, masks_tuned, masks_orig):
    """Returns image of tuned model vs original SAM model

    Args:
        image (_type_): image to be masked
        input_bbox (_type_): bounding box of broken images
        masks_tuned (_type_): tuned sam model
        masks_orig (_type_): tuned original model
    """
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

def plot_train_mean(losses):
    """Plot the train losses

    Args:
        losses (_type_): integer mean loss after training per epoch
    """
    mean_losses = [mean(x) for x in losses]
    mean_losses

    plt.plot(list(range(len(mean_losses))), mean_losses)
    plt.title('Mean epoch loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')

    plt.show()

def compare_models_masks(image, input_bbox, masks_tuned, masks_orig):
    """Return two plot comparison of tuned model vs the original model

    Args:
        image (_type_): image to mask
        input_bbox (_type_): bounding box of the broken bottle to prompt sam
        masks_tuned (_type_): tuned SAM model
        masks_orig (_type_): original SAM model
    """
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



if __name__ == '__main__':
    pass