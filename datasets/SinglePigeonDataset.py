import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import random
import torchvision


class SinglePigeonDataset(Dataset):
    def __init__(self, data_path, session, camera_ids, flip_probability=0.5, scale_percentages_range=None,
                 transform=None, target_transform=None):
        if scale_percentages_range is None:
            scale_percentages_range = [95, 105]
        self.data_path = data_path
        self.session = session
        self.camera_ids = camera_ids
        self.flip_probability = flip_probability
        self.scale_percentages_range = scale_percentages_range
        self.transform = transform
        self.target_transform = target_transform

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
        img_name = 'img_' + str(int(img_labels.iloc[idx, 0] + 1)).zfill(6) + '.jpg'
        img_path = os.path.join(self.data_path, self.session, self.camera_ids[view], img_name)
        image = cv2.imread(img_path)

        ## Flip image
        flip = random.random() < self.flip_probability
        if flip:
            image = cv2.flip(image, 1)
        else:
            pass

        ## Scale image
        # Randomly choose integer in range
        scale_percent = random.randint(self.scale_percentages_range[0], self.scale_percentages_range[1])
        width_scaled = int(image.shape[1] * scale_percent / 100)
        height_scaled = int(image.shape[0] * scale_percent / 100)
        dim = (width_scaled, height_scaled)
        pad_horizontally = int((image.shape[0] - height_scaled) / 2)  # needed only for 'scale_percent < 100'
        pad_vertically = int((image.shape[1] - width_scaled) / 2)  # needed only for 'scale_percent < 100'
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)  # resize image
        crop_vertically = int((resized.shape[1] - image.shape[1]) / 2)  # needed only for 'scale_percent > 100'
        crop_horizontally = int((resized.shape[0] - image.shape[0]) / 2)  # needed only for 'scale_percent > 100'
        if scale_percent == 100:  # no scaling
            del resized
        elif scale_percent < 100:  # smaller
            image = cv2.copyMakeBorder(resized,
                                       top=pad_horizontally,
                                       bottom=pad_horizontally,
                                       left=pad_vertically,
                                       right=pad_vertically,
                                       borderType=cv2.BORDER_REPLICATE)
            del resized
        else:  # larger
            center = [resized.shape[0] / 2, resized.shape[1] / 2]
            x = center[1] - image.shape[1] / 2
            y = center[0] - image.shape[0] / 2
            image = resized[int(y):int(y + image.shape[0]), int(x):int(x + image.shape[1])]
            del resized
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # bgr to rgb

        ### Bbox: FloatTensor[detectedInstances == 1, 4] ###
        if img_labels.iloc[idx, 37] > img_labels.iloc[idx, 39]:
            bbox = np.array([img_labels.iloc[idx, 36], img_labels.iloc[idx, 39],
                    img_labels.iloc[idx, 38], img_labels.iloc[idx, 37]], dtype=np.float32)
        else:
            bbox = np.array(img_labels.iloc[idx, 36:40], dtype=np.float32)
        # Flip
        if flip:
            bbox[0] = - bbox[0] + image.shape[1]
            bbox[2] = - bbox[2] + image.shape[1]
            if bbox[0] > bbox[2]:  # due to flipping x_min can become x_max and vice versa
                bbox_tmp = bbox.copy()
                bbox[0] = bbox_tmp[2]
                bbox[2] = bbox_tmp[0]
                del bbox_tmp
            else:
                pass
        else:
            pass
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
        bbox = torch.from_numpy(bbox)

        ### Keypoints: FloatTensor[detectedInstances == 1, #keypoints, [x, y, visibility]] ###
        keypoints = np.array(img_labels.iloc[idx, 1:15], dtype=np.float32)
        keypoints = np.reshape(keypoints, (int(keypoints.shape[0] / 2), 2))
        # Flip
        if flip:
            keypoints[:, 0] = - keypoints[:, 0] + image.shape[1]
            # flip also 'labels' (=indices in array) of left/right eye/shoulder
            keypoints_tmp = keypoints.copy()
            keypoints[2] = keypoints_tmp[3]
            keypoints[3] = keypoints_tmp[2]
            keypoints[4] = keypoints_tmp[5]
            keypoints[5] = keypoints_tmp[4]
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
                if (keypoints[kp_idx][0] < 0) or (keypoints[kp_idx][0] > image.shape[1])\
                        or (keypoints[kp_idx][1] < 0) or (keypoints[kp_idx][1] > image.shape[0]):
                    keypoints[kp_idx][2] = 0
        else:  # no image scaling
            # Visibility
            # We only use videos where the pigeon is visible all the time -> visibility = 1
            keypoints = np.concatenate((keypoints, np.ones((keypoints.shape[0], 1), dtype=np.float32)), axis=1)
        keypoints = np.reshape(keypoints, (1, keypoints.shape[0], keypoints.shape[1]))
        keypoints = torch.from_numpy(keypoints)

        num_objs = 1  # detectedInstances == 1

        area = torchvision.ops.box_area(bbox)

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {'boxes': bbox,
                  'labels': torch.ones((num_objs,), dtype=torch.int64),  # Int64Tensor[detectedInstances == 1], label: 1
                  'keypoints': keypoints,
                  'image_id': torch.tensor([idx]),
                  'area': area,
                  'iscrowd': iscrowd}

        ### Transforms ###
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target
