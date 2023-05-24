from dataclasses import dataclass

@dataclass
class DataPath:
    ground_truth_masks = "dataset/ground_truth/broken_large"
    train_bottles = 'dataset/train/good'
    test_bottles = 'dataset/test/broken_large/'

@dataclass
class GroundTruth:
    ground_truth_image = '005'