import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from statistics import mean
from collections import defaultdict
from torch.nn.functional import threshold, normalize
from variables import DataPath, GroundTruth
from fine_tuning.config import model_type, checkpoint, device, lr, wd, num_epochs
import sys



