from dataclasses import dataclass

@dataclass
class DataPath:
    ground_truth_masks = "bottle/ground_truth/broken_large/"
    train_bottles = 'bottle/train/good/'
    test_bottles = 'bottle/test/broken_large/'

@dataclass
class GroundTruth:
    ground_truth_image = '005'