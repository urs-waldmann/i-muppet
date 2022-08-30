import cv2
import model_utils
import network_utils
import torchvision
import numpy as np

COLOR_KEYPOINTS = (255, 0, 0)  # bgr
COLOR_LABELS = (255, 0, 0)  # bgr

COLOR_LEFT_SIDE = (128, 128, 240)  # bgr
COLOR_RIGHT_SIDE = (255, 255, 0)  # bgr
COLOR_OTHER_PARTS = (0, 140, 255)  # bgr

POSE_THICKNESS = 2

COLOR_MASK = (128, 0, 128)  # rgb

ID_TEXT_FONT_SCALE = 0.5
ID_TEXT_COLOR = (0, 0, 255)  # bgr
ID_TEXT_THICKNESS = 2


def plot_bbox(cv_image, bbox, bbox_color):

    cv2.rectangle(img=cv_image, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=bbox_color)


def plot_keypoints_with_labels(cv_image, keypoints,
                               plot_labels=False, looking_for_object='pigeon'):

    keypoint_names = model_utils.keypoint_names(
        for_object=looking_for_object
    )

    for j, p in enumerate(keypoints):
        point = (int(p[0]), int(p[1]))
        cv2.drawMarker(
            cv_image,
            point,
            color=COLOR_KEYPOINTS,
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=1
        )
        if plot_labels:
            cv2.putText(
                cv_image,
                keypoint_names[j],
                point,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=COLOR_LABELS
            )


def plot_pose(cv_image, keypoints,
              looking_for_object='pigeon'):

    if looking_for_object == 'pigeon':
        # Left side
        cv2.line(
            cv_image,
            (int(keypoints[1][0]), int(keypoints[1][1])),
            (int(keypoints[2][0]), int(keypoints[2][1])),
            color=COLOR_LEFT_SIDE,
            thickness=POSE_THICKNESS
        )
        cv2.line(
            cv_image,
            (int(keypoints[2][0]), int(keypoints[2][1])),
            (int(keypoints[4][0]), int(keypoints[4][1])),
            color=COLOR_LEFT_SIDE,
            thickness=POSE_THICKNESS
        )
        cv2.line(
            cv_image,
            (int(keypoints[4][0]), int(keypoints[4][1])),
            (int(keypoints[6][0]), int(keypoints[6][1])),
            color=COLOR_LEFT_SIDE,
            thickness=POSE_THICKNESS
        )

        # Right side
        cv2.line(
            cv_image,
            (int(keypoints[1][0]), int(keypoints[1][1])),
            (int(keypoints[3][0]), int(keypoints[3][1])),
            color=COLOR_RIGHT_SIDE,
            thickness=POSE_THICKNESS
        )
        cv2.line(
            cv_image,
            (int(keypoints[5][0]), int(keypoints[5][1])),
            (int(keypoints[3][0]), int(keypoints[3][1])),
            color=COLOR_RIGHT_SIDE,
            thickness=POSE_THICKNESS
        )
        cv2.line(
            cv_image,
            (int(keypoints[5][0]), int(keypoints[5][1])),
            (int(keypoints[6][0]), int(keypoints[6][1])),
            color=COLOR_RIGHT_SIDE,
            thickness=POSE_THICKNESS
        )

        # 'Beak'
        cv2.line(
            cv_image,
            (int(keypoints[0][0]), int(keypoints[0][1])),
            (int(keypoints[1][0]), int(keypoints[1][1])),
            color=COLOR_OTHER_PARTS,
            thickness=POSE_THICKNESS
        )
    elif looking_for_object == 'person':
        # 'Head'
        cv2.line(
            cv_image,
            (int(keypoints[0][0]), int(keypoints[0][1])),
            (int(keypoints[1][0]), int(keypoints[1][1])),
            color=COLOR_OTHER_PARTS,
            thickness=POSE_THICKNESS
        )
        cv2.line(
            cv_image,
            (int(keypoints[0][0]), int(keypoints[0][1])),
            (int(keypoints[2][0]), int(keypoints[2][1])),
            color=COLOR_OTHER_PARTS,
            thickness=POSE_THICKNESS
        )
        cv2.line(
            cv_image,
            (int(keypoints[3][0]), int(keypoints[3][1])),
            (int(keypoints[1][0]), int(keypoints[1][1])),
            color=COLOR_OTHER_PARTS,
            thickness=POSE_THICKNESS
        )
        cv2.line(
            cv_image,
            (int(keypoints[4][0]), int(keypoints[4][1])),
            (int(keypoints[2][0]), int(keypoints[2][1])),
            color=COLOR_OTHER_PARTS,
            thickness=POSE_THICKNESS
        )

        # 'Arms'
        cv2.line(
            cv_image,
            (int(keypoints[5][0]), int(keypoints[5][1])),
            (int(keypoints[7][0]), int(keypoints[7][1])),
            color=COLOR_LEFT_SIDE,
            thickness=POSE_THICKNESS
        )
        cv2.line(
            cv_image,
            (int(keypoints[9][0]), int(keypoints[9][1])),
            (int(keypoints[7][0]), int(keypoints[7][1])),
            color=COLOR_LEFT_SIDE,
            thickness=POSE_THICKNESS
        )
        cv2.line(
            cv_image,
            (int(keypoints[6][0]), int(keypoints[6][1])),
            (int(keypoints[8][0]), int(keypoints[8][1])),
            color=COLOR_LEFT_SIDE,
            thickness=POSE_THICKNESS
        )
        cv2.line(
            cv_image,
            (int(keypoints[8][0]), int(keypoints[8][1])),
            (int(keypoints[10][0]), int(keypoints[10][1])),
            color=COLOR_LEFT_SIDE,
            thickness=POSE_THICKNESS
        )

        # 'Legs'
        cv2.line(
            cv_image,
            (int(keypoints[11][0]), int(keypoints[11][1])),
            (int(keypoints[13][0]), int(keypoints[13][1])),
            color=COLOR_RIGHT_SIDE,
            thickness=POSE_THICKNESS
        )
        cv2.line(
            cv_image,
            (int(keypoints[14][0]), int(keypoints[14][1])),
            (int(keypoints[12][0]), int(keypoints[12][1])),
            color=COLOR_RIGHT_SIDE,
            thickness=POSE_THICKNESS
        )
        cv2.line(
            cv_image,
            (int(keypoints[14][0]), int(keypoints[14][1])),
            (int(keypoints[16][0]), int(keypoints[16][1])),
            color=COLOR_RIGHT_SIDE,
            thickness=POSE_THICKNESS
        )
        cv2.line(
            cv_image,
            (int(keypoints[15][0]), int(keypoints[15][1])),
            (int(keypoints[13][0]), int(keypoints[13][1])),
            color=COLOR_RIGHT_SIDE,
            thickness=POSE_THICKNESS
        )
    elif looking_for_object == 'mouse':
        # Left side
        cv2.line(
            cv_image,
            (int(keypoints[0][0]), int(keypoints[0][1])),
            (int(keypoints[1][0]), int(keypoints[1][1])),
            color=COLOR_LEFT_SIDE,
            thickness=POSE_THICKNESS
        )
        cv2.line(
            cv_image,
            (int(keypoints[1][0]), int(keypoints[1][1])),
            (int(keypoints[3][0]), int(keypoints[3][1])),
            color=COLOR_LEFT_SIDE,
            thickness=POSE_THICKNESS
        )

        # Right side
        cv2.line(
            cv_image,
            (int(keypoints[0][0]), int(keypoints[0][1])),
            (int(keypoints[2][0]), int(keypoints[2][1])),
            color=COLOR_RIGHT_SIDE,
            thickness=POSE_THICKNESS
        )
        cv2.line(
            cv_image,
            (int(keypoints[2][0]), int(keypoints[2][1])),
            (int(keypoints[3][0]), int(keypoints[3][1])),
            color=COLOR_RIGHT_SIDE,
            thickness=POSE_THICKNESS
        )


def plot_mask(cv_image, bool_mask, opacity=0.5):

    rgb_tensor = network_utils.image_cv_to_rgb_tensor(cv_image, scaling=False)
    rgb_with_mask = torchvision.utils.draw_segmentation_masks(
        rgb_tensor,
        bool_mask,
        alpha=opacity,
        colors=COLOR_MASK
    )
    # Convert to numpy array
    rgb_with_mask = rgb_with_mask.numpy()
    cv_image_with_mask = np.transpose(rgb_with_mask, (1, 2, 0)).copy()
    cv_image_with_mask = cv2.cvtColor(cv_image_with_mask, cv2.COLOR_RGB2BGR)

    return cv_image_with_mask


def plot_id(cv_image, objects=None, track_bbs_ids=None):

    if objects is not None:
        # loop over the tracked objects
        for objectID, centroid in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(
                cv_image,
                text,
                # draw ID of the object in the middle of the bbox
                (centroid[0], centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                ID_TEXT_FONT_SCALE,
                ID_TEXT_COLOR,
                ID_TEXT_THICKNESS
            )

    if track_bbs_ids is not None:
        # loop over the tracked objects
        for instance in track_bbs_ids:
            text = "ID {}".format(int(instance[4] - 1))
            cv2.putText(
                cv_image,
                text,
                # draw ID of the object in the middle of the bbox
                (int((instance[0] + instance[2]) / 2), int((instance[1] + instance[3]) / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                ID_TEXT_FONT_SCALE,
                ID_TEXT_COLOR,
                ID_TEXT_THICKNESS
            )
