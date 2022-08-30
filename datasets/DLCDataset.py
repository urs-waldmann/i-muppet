import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import random
import torchvision


class DLCDataset(Dataset):
    def __init__(self, data_path, session, camera_ids, flip_probability=0.5, scale_percentages_range=None,
                 scale_bbox=0.5, transform=None, target_transform=None):
        if scale_percentages_range is None:
            scale_percentages_range = [95, 105]
        self.data_path = data_path
        self.session = session
        self.camera_ids = camera_ids
        self.flip_probability = flip_probability
        self.scale_percentages_range = scale_percentages_range
        self.transform = transform
        self.target_transform = target_transform
        self.scale_bbox = scale_bbox

    def __len__(self):
        # All .csv files/views have same number of labeled data for one session
        # -> I choose one randomly -> camera_ids[0]
        img_labels = pd.read_csv(os.path.join(self.data_path, self.session, self.session + '.' + self.camera_ids[0]
                                              + '.' + 'database.csv'))

        return len(img_labels)

    def __getitem__(self, idx):

        # Choose randomly between views/camera_ids
        view = random.randint(0, len(self.camera_ids) - 1)

        # Load labels of random view/camera_id
        img_labels = pd.read_csv(os.path.join(self.data_path, self.session, self.session + '.' + self.camera_ids[view]
                                              + '.' + 'database.csv'))

        ### Image ###
        # First frame number in .csv file is 0, first image number is 1
        img_name = 'img_' + str(int(img_labels.iloc[idx, 10] + 1)).zfill(6) + '.png'
        img_path = os.path.join(self.data_path, self.session, self.camera_ids[view], img_name)
        image = cv2.imread(img_path)
        image_one_size = cv2.resize(image, (800, 800), interpolation=cv2.INTER_LINEAR)

        ## Flip image
        flip = random.random() < self.flip_probability
        if flip:
            image_one_size = cv2.flip(image_one_size, 1)
        else:
            pass

        ## Scale image
        # Randomly choose integer in range
        scale_percent = random.randint(self.scale_percentages_range[0], self.scale_percentages_range[1])
        width_scaled = int(image_one_size.shape[1] * scale_percent / 100)
        height_scaled = int(image_one_size.shape[0] * scale_percent / 100)
        dim = (width_scaled, height_scaled)
        pad_horizontally = int((image_one_size.shape[0] - height_scaled) / 2)  # needed only for 'scale_percent < 100'
        pad_vertically = int((image_one_size.shape[1] - width_scaled) / 2)  # needed only for 'scale_percent < 100'
        resized = cv2.resize(image_one_size, dim, interpolation=cv2.INTER_LINEAR)  # resize image
        crop_vertically = int((resized.shape[1] - image_one_size.shape[1]) / 2)  # needed only for 'scale_percent > 100'
        # needed only for 'scale_percent > 100'
        crop_horizontally = int((resized.shape[0] - image_one_size.shape[0]) / 2)
        if scale_percent == 100:  # no scaling
            del resized
        elif scale_percent < 100:  # smaller
            image_one_size = cv2.copyMakeBorder(resized,
                                       top=pad_horizontally,
                                       bottom=pad_horizontally,
                                       left=pad_vertically,
                                       right=pad_vertically,
                                       borderType=cv2.BORDER_REPLICATE)
            del resized
        else:  # larger
            center = [resized.shape[0] / 2, resized.shape[1] / 2]
            x = center[1] - image_one_size.shape[1] / 2
            y = center[0] - image_one_size.shape[0] / 2
            image_one_size = resized[int(y):int(y + image_one_size.shape[0]), int(x):int(x + image_one_size.shape[1])]
            del resized
        image_one_size = cv2.cvtColor(image_one_size, cv2.COLOR_BGR2RGB)  # bgr to rgb

        ### Keypoints: FloatTensor[detectedInstances == 1, #keypoints, [x, y, visibility]] ###
        keypoints = np.array(img_labels.iloc[idx, 1:9], dtype=np.float32)
        keypoints = np.reshape(keypoints, (int(keypoints.shape[0] / 2), 2))
        # Scale keypoints due to one size 800 x 800
        if image_one_size.shape != image.shape:
            factor_x = image_one_size.shape[1] / image.shape[1]
            factor_y = image_one_size.shape[0] / image.shape[0]
            keypoints[:, 0] *= factor_x
            keypoints[:, 1] *= factor_y
        # Flip
        if flip:
            keypoints[:, 0] = - keypoints[:, 0] + image_one_size.shape[1]
            # flip also 'labels' (=indices in array) of left/right ear
            keypoints_tmp = keypoints.copy()
            keypoints[1] = keypoints_tmp[2]
            keypoints[2] = keypoints_tmp[1]
            del keypoints_tmp
        else:
            pass
        # Scaling
        if scale_percent < 100:  # image scaled smaller
            keypoints = keypoints * (scale_percent / 100)  # scale keypoints to resized image
            keypoints[:, 0] = keypoints[:, 0] + pad_vertically  # scale keypoints to padded image
            keypoints[:, 1] = keypoints[:, 1] + pad_horizontally
            # Visibility
            # We only use videos where the pigeon is visible all the time -> visibility = 1
            keypoints = np.concatenate((keypoints, np.ones((keypoints.shape[0], 1), dtype=np.float32)), axis=1)
        elif scale_percent > 100:  # image scaled larger
            keypoints = keypoints * (scale_percent / 100)  # scale keypoints to resized image
            keypoints[:, 0] = keypoints[:, 0] - crop_vertically  # scale keypoints to cropped image
            keypoints[:, 1] = keypoints[:, 1] - crop_horizontally
            # Visibility
            # We only use videos where the pigeon is visible all the time.
            # But if image is scaled larger we can crop the image s.t. the pigeon is not visible anymore or only partly
            # -> visibility = 0 for all/some keypoints
            keypoints = np.concatenate((keypoints, np.ones((keypoints.shape[0], 1), dtype=np.float32)), axis=1)
            for kp_idx in range(keypoints.shape[0]):
                if (keypoints[kp_idx][0] < 0) or (keypoints[kp_idx][0] > image_one_size.shape[1])\
                        or (keypoints[kp_idx][1] < 0) or (keypoints[kp_idx][1] > image_one_size.shape[0]):
                    keypoints[kp_idx][2] = 0
        else:  # no image scaling
            # Visibility
            # We only use videos where the pigeon is visible all the time -> visibility = 1
            keypoints = np.concatenate((keypoints, np.ones((keypoints.shape[0], 1), dtype=np.float32)), axis=1)
        keypoints = np.reshape(keypoints, (1, keypoints.shape[0], keypoints.shape[1]))
        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        ### Bbox: FloatTensor[detectedInstances == 1, 4] ###
        x_min = np.amin(keypoints.numpy()[:, :, 0])
        x_max = np.amax(keypoints.numpy()[:, :, 0])
        y_min = np.amin(keypoints.numpy()[:, :, 1])
        y_max = np.amax(keypoints.numpy()[:, :, 1])

        # Scale bbox to fit whole mouse (we used min/max keypoints only to create bbox)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        if bbox_height <= bbox_width:
            bbox_length = bbox_width
        else:
            bbox_length = bbox_height
        x_min = x_min - (self.scale_bbox / 2) * bbox_length
        x_max = x_max + (self.scale_bbox / 2) * bbox_length
        y_min = y_min - (self.scale_bbox / 2) * bbox_length
        y_max = y_max + (self.scale_bbox / 2) * bbox_length
        if x_min < 0:
            x_min = 0
        elif x_max > image_one_size.shape[1]:
            x_max = image_one_size.shape[1] - 1
        elif y_min < 0:
            y_min = 0
        elif y_max > image_one_size.shape[0]:
            y_max = image_one_size.shape[0] - 1
        else:
            pass

        bbox = np.array([x_min, y_min, x_max, y_max])

        # Flip
        # No flipping needed since we used flipped keypoints to create bbox

        # Scaling
        if scale_percent < 100:  # image scaled smaller
            bbox = bbox * (scale_percent / 100)  # scale bbox to resized image
            bbox[0] = bbox[0] + pad_vertically  # scale bbox to padded image
            bbox[2] = bbox[2] + pad_vertically
            bbox[1] = bbox[1] + pad_horizontally
            bbox[3] = bbox[3] + pad_horizontally
        elif scale_percent > 100:  # image scaled larger
            bbox = bbox * (scale_percent / 100)  # scale bbox to resized image
            bbox[0] = bbox[0] - crop_vertically  # scale bbox to cropped image
            bbox[2] = bbox[2] - crop_vertically
            bbox[1] = bbox[1] - crop_horizontally
            bbox[3] = bbox[3] - crop_horizontally
        else:  # no image scaling
            pass
        bbox = np.reshape(bbox, (1, 4))
        bbox = torch.tensor(bbox, dtype=torch.float32)

        num_objs = 1  # detectedInstances == 1

        area = torchvision.ops.box_area(bbox)

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            'boxes': bbox,
            'labels': torch.ones((num_objs,), dtype=torch.int64),  # Int64Tensor[detectedInstances == 1], label: 1
            'keypoints': keypoints,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': iscrowd
        }

        ### Transforms ###
        if self.transform:
            image_one_size = self.transform(image_one_size)
        if self.target_transform:
            target = self.target_transform(target)

        return image_one_size, target
