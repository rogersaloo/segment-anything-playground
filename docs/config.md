# Configurations for Fine-tuning

## Description
Contains the hyperparameters tuned for training the samp model on a specific task.

### Hyper-params
1. Learning rate to 1e-06 
2. Change image decoder loss type to MSE
3. Using the Adam optimizer on the image decoder

### Variables 
1. model_type = 'vit_b'
2. checkpoint = 'model_checkpoint/sam_vit_b_01ec64.pth'
3. device = 'cuda:0'

### Path variables
1. ground_truth_masks = "dataset/ground_truth/broken_large" location of the ground truth images of broken bottles.
2. train_bottles = 'dataset/train/good' - location of the normal and broken bottles used for training sam model.
3. test_bottles = 'dataset/test/broken_large/' - location of images used to test the fine-tuned sam model.