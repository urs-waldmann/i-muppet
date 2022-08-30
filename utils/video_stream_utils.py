import cv2
import torch
import network_utils
import vis_utils
import os
import misc_utils
import model_utils


def initialize_video_stream(video_stream):

    print("[INFO] initializing video stream...")

    return cv2.VideoCapture(video_stream)


def process_video_stream(video_stream, network, instance_category_names, confidence_score, tracker,
                         plot_detector_bbox, plot_keypoints, plot_labels, plot_pose, plot_mask, threshold_masks,
                         trained_model, opacity,
                         device=torch.device('cpu'), plot_id=True, plot_tracker_bbox=True, save_images=False,
                         image_folder=None, looking_for_object='pigeon', full_screen=False, write_tracking_data=False,
                         tracking_data_file_name='predictions.txt', tracking_data_folder=None, challenge=None):

    # save the output frames
    frame_count = 1
    if save_images:
        if image_folder is None:
            image_folder = './data/imgs'
        if looking_for_object == 'person':
            image_folder = os.path.join(image_folder,
                                        video_stream[video_stream.rfind('/') + 1:-4],
                                        'person_model_basic'
                                        + '_cs_'
                                        + str(confidence_score)[0]
                                        + str(confidence_score)[str(confidence_score).rfind('.') + 1:]
                                        + '_' + tracker.__class__.__name__)
        else:
            image_folder = os.path.join(image_folder,
                                        video_stream[video_stream.rfind('/') + 1:-4],
                                        trained_model[trained_model.rfind('/') + 1:-3]
                                        + '_cs_'
                                        + str(confidence_score)[0]
                                        + str(confidence_score)[str(confidence_score).rfind('.') + 1:]
                                        + '_' + tracker.__class__.__name__)
        misc_utils.create_folder(image_folder)

    # write the tracking data
    if write_tracking_data:
        if (tracking_data_folder is None) or (challenge is None):
            path_det = os.path.join('./data/tracking/gt',
                                  tracking_data_file_name[:-4],
                                  'det')
            path_pred = './data/tracking/trackers/iMuppet/data'
            misc_utils.create_folder(path_det)
            misc_utils.create_folder(path_pred)
            txt_file_detector = open(os.path.join(path_det,
                                                  'det.txt'),
                                     'w')
            txt_file_tracker = open(os.path.join(path_pred,
                                                 tracking_data_file_name),
                                    'w')
        else:
            txt_file_detector = open(os.path.join(tracking_data_folder,
                                                  'gt',
                                                  challenge,
                                                  tracking_data_file_name[:-4],
                                                  'det/det.txt'),
                                     'w')
            txt_file_tracker = open(os.path.join(tracking_data_folder,
                                                 'trackers',
                                                 challenge,
                                                 'iMuppet/data',
                                                 tracking_data_file_name),
                                    'w')

    # initialize the video stream
    cap = initialize_video_stream(
        video_stream=video_stream
    )

    print("[INFO] processing video stream...")

    while True:
        is_success, cv_image = cap.read()

        if not is_success:
            break

        # prepare one image for inference
        image = network_utils.image_cv_to_rgb_tensor(
            cv_image=cv_image
        )
        if model_utils.is_coco_instance(looking_for_object, network._get_name()):
            pass
        else:
            image = network_utils.normalize_tensor_image(
                tensor_image=image,
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        image = network_utils.image_to_device(image=image, device=device)

        # inference
        result = network_utils.infer_one_image(
            network=network,
            image=image
        )

        # transform predictions
        scores, labels, boxes, keypoints, masks = network_utils.transform_predictions(
            result
        )

        # process all detected instances
        detections, cv_image = network_utils.process_all_detected_instances(
            cv_image=cv_image,
            scores=scores,
            labels=labels,
            bboxes=boxes,
            keypoints=keypoints,
            masks=masks,
            instance_category_names=instance_category_names,
            looking_for_object=looking_for_object,
            confidence_score=confidence_score,
            threshold_masks=threshold_masks,
            tracker=tracker,
            plot_detector_bbox=plot_detector_bbox,
            plot_keypoints=plot_keypoints,
            plot_labels=plot_labels,
            plot_pose=plot_pose,
            plot_mask=plot_mask,
            opacity=opacity
        )

        # write detections data for tracking evaluation
        if write_tracking_data:
            # # write only frames that are also present in ground truth
            # if frame_count in gt_frame_indices:
            for obj in detections:
                txt_file_detector.write('%d,' % frame_count)  # frame number
                txt_file_detector.write('%d,' % -1)  # object index
                # bbox
                x_min, y_min, x_max, y_max = obj[:4]
                bb_width = x_max - x_min  # format for MOT challenge
                bb_height = y_max - y_min  # format for MOT challenge
                assert bb_width >= 0
                assert bb_height >= 0
                txt_file_detector.write('%d,' % x_min)
                txt_file_detector.write('%d,' % y_min)
                txt_file_detector.write('%d,' % bb_width)
                txt_file_detector.write('%d,' % bb_height)
                txt_file_detector.write("{:.3f}".format(obj[4] * 100))  # detection confidence score
                txt_file_detector.write(',')
                txt_file_detector.write('%d,' % -1)  # ignore 3D coordinates
                txt_file_detector.write('%d,' % -1)  # ignore 3D coordinates
                txt_file_detector.write('%d' % -1)  # ignore 3D coordinates
                txt_file_detector.write('\n')  # new line

        # update the tracker using the processed instances, plot ID (and write tracking data)
        # Centroid Tracker
        if tracker.__class__.__name__ == 'CentroidTracker':
            detections = [detections[i][:4] for i in range(len(detections))]  # remove score
            detections = [list(map(int, detections[i])) for i in range(len(detections))]  # convert to integer

            objects = tracker.update(detections)

            # Plot ID
            if plot_id:
                vis_utils.plot_id(
                    cv_image=cv_image,
                    objects=objects
                )
        # SORT
        elif tracker.__class__.__name__ == 'Sort':
            track_bbs_ids = tracker.update(detections)
            # Plot ID & bboxes
            if plot_id:
                vis_utils.plot_id(
                    cv_image=cv_image,
                    track_bbs_ids=track_bbs_ids
                )
            # also plot updated bboxes
            if plot_tracker_bbox:
                # loop over the tracked objects
                for instance in track_bbs_ids:
                    vis_utils.plot_bbox(
                        cv_image=cv_image,
                        bbox=[int(instance[0]), int(instance[1]), int(instance[2]), int(instance[3])],
                        bbox_color=(0, 0, 255)
                    )
            # write tracking data
            if write_tracking_data:
                # # write only frames that are also present in ground truth
                # if frame_count in gt_frame_indices:
                for obj in track_bbs_ids:
                    txt_file_tracker.write('%d,' % frame_count)  # frame number
                    txt_file_tracker.write('%d,' % obj[4])  # object index
                    # bbox
                    x_min, y_min, x_max, y_max = obj[:4]
                    bb_width = x_max - x_min  # format for MOT challenge
                    bb_height = y_max - y_min  # format for MOT challenge
                    assert bb_width >= 0
                    assert bb_height >= 0
                    txt_file_tracker.write('%d,' % x_min)
                    txt_file_tracker.write('%d,' % y_min)
                    txt_file_tracker.write('%d,' % bb_width)
                    txt_file_tracker.write('%d,' % bb_height)
                    txt_file_tracker.write('%d,' % 1)  # use detection confidence score
                    txt_file_tracker.write('%d,' % -1)  # ignore 3D coordinates
                    txt_file_tracker.write('%d,' % -1)  # ignore 3D coordinates
                    txt_file_tracker.write('%d' % -1)  # ignore 3D coordinates
                    txt_file_tracker.write('\n')  # new line
        else:
            assert False

        # save the output frame
        if save_images:
            image_name = 'frame_' + str(frame_count) + '.jpg'
            cv2.imwrite(os.path.join(image_folder, image_name), cv_image)

        frame_count += 1

        # show the output frame
        window_name = 'Tracking'
        if full_screen:
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, cv_image)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    print('[INFO] finished')

    # close tracking data files
    if write_tracking_data:
        txt_file_detector.close()
        txt_file_tracker.close()
