import os
import sys
sys.path.insert(0, os.path.abspath('./datasets'))
from cowbird_dataset import Cowbird_Dataset
from DLCDataset import DLCDataset
from SinglePigeonDataset import SinglePigeonDataset


def load_dataset(transform,
                 species='pigeon', session='united_sessions_muppet_600', flip_probability=0.5,
                 scale_percentages_range=None):

    if scale_percentages_range is None:
        scale_percentages_range = [100, 100]
    if species == 'pigeon':
        dataset = SinglePigeonDataset(data_path='./data/annotations/pigeon_data',
                                      session=session,
                                      camera_ids=['united_cameras'],
                                      flip_probability=flip_probability,
                                      scale_percentages_range=scale_percentages_range,
                                      transform=transform,
                                      target_transform=None)
    elif species == 'cowbird':
        dataset = Cowbird_Dataset(root='./data/annotations/cowbird/images',
                                  annfile='./data/annotations/cowbird/annotations/instance_train.json',
                                  transform=transform)
    elif species == 'mouse':
        dataset = DLCDataset(data_path='./data/annotations/dlc_data',
                             session=session,
                             camera_ids=['united_cameras'],
                             flip_probability=flip_probability,
                             scale_percentages_range=scale_percentages_range,
                             scale_bbox=0.5,
                             transform=transform,
                             target_transform=None)
    else:
        assert False

    return dataset
