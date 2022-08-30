### Load modules
import os
import yaml
from easydict import EasyDict as Edict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def create_folder(path):

    """
    Creates folder if it does not exist
    """

    if not os.path.exists(path):
        os.makedirs(path)


def load_config(path):

    """
    Loads configuration file.
    """

    with open(path) as fin:
        config = Edict(yaml.safe_load(fin))

    return config


def display_dataset(dataloader, species, check_left_right=False, alpha=0.4):

    """
    Displays some frames of data set with its targets.

    Parameters
    - check_left_right: Set True to display left and right keypoint labels.
    - alpha: Set opacity of the mask overlay.
    """

    print("[INFO] display data...")

    train_features, train_labels = next(iter(dataloader))
    print(f"\nBatch size: {len(train_features)}")
    print(f"Feature shape: {train_features[0].size()}")
    print(f"Boxes shape: {train_labels[0]['boxes'].size()}")
    print(f"Labels shape: {train_labels[0]['labels'].size()}")
    if species in ['pigeon', 'mouse', 'cowbird']:
        print(f"Keypoints shape: {train_labels[0]['keypoints'].size()}")
    elif species == 'macaque':
        print(f"Masks shape: {train_labels[0]['masks'].size()}")
    else:
        assert False
    print(f"Label: {train_labels[0]['labels'][0].item()}\n")
    ax = []
    fig = plt.figure(figsize=(32, 32))
    columns = 3
    rows = 3
    for i in range(1, columns * rows + 1):
        img = np.transpose(train_features[2 * (i - 1)].numpy(), (1, 2, 0)).copy()
        labels = train_labels[2 * (i - 1)]
        ax.append(fig.add_subplot(rows, columns, i))
        if species in ['pigeon', 'mouse', 'cowbird']:
            plt.imshow(img)
        elif species == 'macaque':
            plt.imshow(img)
            # loop over detectedInstances
            for mask in labels['masks']:
                plt.imshow(mask, cmap='viridis', interpolation='none', alpha=alpha)
        else:
            assert False
        if species in ['pigeon', 'mouse', 'cowbird']:
            for kp in range(labels['keypoints'].size()[1]):
                # keypoint color according to visibility
                kp_color = 'dodgerblue' if labels['keypoints'][0, kp, 2].item() == 1 else 'red'
                ax[i - 1].scatter(labels['keypoints'][0, kp, 0:2].numpy()[0],
                                  labels['keypoints'][0, kp, 0:2].numpy()[1],
                                  s=1,
                                  c=kp_color)
                if check_left_right:
                    if species == 'pigeon':
                        if kp in [2, 4]:
                            ax[i - 1].annotate('L', (labels['keypoints'][0, kp, 0:2].numpy()[0],
                                                     labels['keypoints'][0, kp, 0:2].numpy()[1]), c='red')
                        if kp in [3, 5]:
                            ax[i - 1].annotate('R', (labels['keypoints'][0, kp, 0:2].numpy()[0],
                                                     labels['keypoints'][0, kp, 0:2].numpy()[1]), c='red')
                    elif species == 'mouse':
                        if kp in [1]:
                            ax[i - 1].annotate('L', (labels['keypoints'][0, kp, 0:2].numpy()[0],
                                                     labels['keypoints'][0, kp, 0:2].numpy()[1]), c='red')
                        if kp in [2]:
                            ax[i - 1].annotate('R', (labels['keypoints'][0, kp, 0:2].numpy()[0],
                                                     labels['keypoints'][0, kp, 0:2].numpy()[1]), c='red')
                    else:
                        assert False
        # loop over detectedInstances
        for box in labels['boxes']:
            ax[i - 1].add_patch(
                Rectangle((box.numpy()[0], box.numpy()[1]),
                          box.numpy()[2] - box.numpy()[0],
                          box.numpy()[3] - box.numpy()[1],
                          linewidth=1, edgecolor='g', facecolor='none'))
    plt.show()
