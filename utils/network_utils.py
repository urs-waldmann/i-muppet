import torchvision
import torch
from torchvision import transforms
import cv2
import numpy as np
import vis_utils
import model_utils


def load_network(network_name='KeypointRCNN', looking_for_object='pigeon', eval_mode=False, pre_trained_model=None,
                 device=torch.device('cpu')):

    print("[INFO] loading network...")

    if network_name == 'KeypointRCNN':
        net = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            pretrained=model_utils.is_coco_instance(
                for_object=looking_for_object,
                network_name=network_name
            ),  # pretrained on COCO train2017
            progress=True,
            num_classes=2,  # number of output classes including background
            pretrained_backbone=True,  # backbone pretrained on Imagenet
            trainable_backbone_layers=0,
            num_keypoints=len(
                model_utils.keypoint_names(
                    for_object=looking_for_object
                ))
        )
        # load (pre)trained model
        if pre_trained_model is not None:
            if looking_for_object == 'person':
                pass
            else:
                print('[INFO] Load pre-trained weights...')
                net.load_state_dict(torch.load(pre_trained_model))
    elif network_name == 'MaskRCNN':
        net = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True,  # pretrained on COCO train2017
            progress=True,
            num_classes=91,  # number of output classes including background
            pretrained_backbone=True,  # backbone pretrained on Imagenet
            trainable_backbone_layers=0
        )
        # do not load (pre)trained model since I do not have any right now
    else:
        print('[ERROR] !!! select available network !!!')
        assert False
    net.to(device)
    if eval_mode:
        net.eval()

    return net


def image_cv_to_rgb_tensor(cv_image, scaling=True):

    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  # bgr to rgb
    if scaling:  # [0, 255] scaled to [0, 1]
        image = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )(image)
    else:  # RGB tensor with data_type = uint8
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)

    return image


def normalize_tensor_image(tensor_image, mean, std):
    image = transforms.Compose(
        [
            transforms.Normalize(mean, std)
        ]
    )(tensor_image)

    return image


def image_to_device(image, device):

    return image.to(device)


def infer_one_image(network, image):

    with torch.inference_mode():
        return network([image])[0]


def transform_predictions(predictions):

    bboxes = predictions['boxes']
    labels = predictions['labels']
    if 'keypoints' in list(predictions):
        keypoints = predictions['keypoints']  # [N,K,3]
    else:
        keypoints = None
    scores = predictions['scores']
    if 'masks' in list(predictions):
        masks = predictions['masks']  # [N, 1, H, W]
    else:
        masks = None

    return scores, labels, bboxes, keypoints, masks


def process_all_detected_instances(cv_image, tracker, scores, labels, bboxes, keypoints, masks, instance_category_names,
                                   opacity,
                                   looking_for_object='pigeon', confidence_score=0.5, plot_detector_bbox=True,
                                   plot_keypoints=False, plot_labels=False, plot_pose=True, plot_mask=True,
                                   threshold_masks=0.5):

    # Initialize list for bboxes
    rects = []

    for i in range(len(labels)):

        label = labels[i].item()
        object_class = instance_category_names[label]
        score = scores[i].item()
        # print(f'Class: {object_class}, score: {score:.04f}')

        # Check if detected instance is a valid instance
        if object_class != looking_for_object:
            continue
        if score <= confidence_score:
            continue

        #print(f'Object of Interest: {object_class}, Score: {score:.04f}')

        ## Bboxes with score
        x1, y1, x2, y2 = bboxes[i, ...].to(dtype=torch.int32)

        box = np.array([x1.item(), y1.item(), x2.item(), y2.item(), score])
        rects.append(box)

        # Plot bboxes from detector
        if plot_detector_bbox:
            vis_utils.plot_bbox(
                cv_image=cv_image,
                bbox=[x1.item(), y1.item(), x2.item(), y2.item()],
                bbox_color=(0, 255, 0)  # bgr
            )

        ## Keypoints
        if keypoints is not None:
            points = keypoints[i, ...].to(dtype=torch.int32)

            # Plot keypoints with labels
            if plot_keypoints:
                vis_utils.plot_keypoints_with_labels(
                    cv_image=cv_image,
                    looking_for_object=looking_for_object,
                    keypoints=points,
                    plot_labels=plot_labels
                )

            # Plot pose
            if plot_pose:
                vis_utils.plot_pose(
                    cv_image=cv_image,
                    looking_for_object=looking_for_object,
                    keypoints=points
                )

        ## Masks
        if masks is not None:
            bool_mask = masks[i] > threshold_masks  # Threshold mask
            bool_mask = bool_mask.squeeze(1)  # There's an extra dimension (1) to the masks. We need to remove it.

            # Plot mask
            if plot_mask:
                cv_image = vis_utils.plot_mask(
                    cv_image=cv_image,
                    bool_mask=bool_mask,
                    opacity=opacity
                )

    if tracker.__class__.__name__ == 'CentroidTracker':
        processed_detections = rects
    elif tracker.__class__.__name__ == 'Sort':
        if len(rects) == 0:  # if there is no detected instance
            processed_detections = np.empty((0, 5))
        else:
            processed_detections = np.array(rects)
    else:
        assert False

    return processed_detections, cv_image
