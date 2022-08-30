import argparse
import torch
import sys
import os
sys.path.insert(0, os.path.abspath('./utils'))
import tracker_utils
import network_utils
import video_stream_utils
import model_utils


def parse_args():

	parser = argparse.ArgumentParser()

	parser.add_argument("--confidence_score",
						type=float,
						default=0.9,
						help="Detector: Set the threshold for the detector confidence score.")
	parser.add_argument("--species",
						type=str,
						default="pigeon",
						help="Input: Specify the object of interest.")
	parser.add_argument("--weights",
						type=str,
						default="muppet",
						help="Input: Specify the trained model.")
	parser.add_argument("--video",
						type=str,
						default="2p_2118670.avi",
						help="Input: Specify the video name (with extension) to process.")
	parser.add_argument("--full_screen",
						action='store_true',
						help="Visualization: Show output video in full screen mode.")
	parser.add_argument("--plot_detector_bbox",
						action='store_true',
						help="Visualization: Plot detector bounding box on the output video.")
	parser.add_argument("--plot_tracker_bbox",
						action='store_true',
						help="Visualization: Plot tracker bounding box on the output video.")
	parser.add_argument("--plot_keypoints",
						action='store_true',
						help="Visualization: Plot keypoints on the output video.")
	parser.add_argument("--plot_labels",
						action='store_true',
						help="Visualization: Plot keypoint labels on the output video (iff --plot_keypoints is 'True').")
	parser.add_argument("--plot_pose",
						action='store_true',
						help="Visualization: Plot pose on the output video.")
	parser.add_argument("--plot_id",
						action='store_true',
						help="Visualization: Plot object ID on the output video.")
	parser.add_argument("--write_tracking_data",
						action='store_true',
						help="Misc.: Write tracking data to TXT file.")
	parser.add_argument("--save_images",
						action='store_true',
						help="Misc.: Save output images to folder.")
	parser.add_argument("--max_age",
						type=int,
						default=1,
						help="Tracker: Maximum number of frames to keep alive a track without associated detections.")
	parser.add_argument("--min_hits",
						type=int,
						default=3,
						help="Tracker: Minimum number of associated detections before track is initialised.")
	parser.add_argument("--iou_threshold",
						type=float,
						default=0.3,
						help="Tracker: Minimum IOU for match.")

	args = parser.parse_args()

	return args


def main(args):

	##########
	# Variables
	# tracker
	tracker = 'sort'  # ['ct', 'sort']
	# ct
	max_disappeared = 1  # default: max_disappeared=50
	# sort
	max_age = args.max_age  # Default: max_age = 1
	min_hits = args.min_hits  # Default: min_hits = 3
	iou_threshold = args.iou_threshold  # Default: iou_threshold = 0.3

	# detector
	network_name = 'KeypointRCNN'  # ['KeypointRCNN', 'MaskRCNN']
	confidence_score = args.confidence_score
	# Mask R-CNN
	threshold_masks = 0.5

	# input
	ooi = args.species  # Object of Interest: ['pigeon', 'person', 'bird', 'mouse']
	trained_model = os.path.join('./data/weights/', args.weights + '.pt')
	video = os.path.join('./data/videos/', args.video)  # 0 for webcam

	# visualization
	full_screen = args.full_screen

	plot_detector_bbox = args.plot_detector_bbox
	plot_tracker_bbox = args.plot_tracker_bbox  # only for SORT tracker

	plot_keypoints = args.plot_keypoints
	if args.plot_keypoints is True:
		plot_labels = args.plot_labels  # plotted iff plot_keypoints = True
	else:
		plot_labels = False

	plot_pose = args.plot_pose

	plot_mask = True
	opacity = 0.3

	plot_id = args.plot_id

	# output
	save_images = args.save_images
	image_folder = None

	# only for SORT tracker
	write_tracking_data = args.write_tracking_data
	tracking_data_folder = None
	challenge = None
	tracking_data_file_name = args.video[:args.video.rfind('.')] + '.txt'
	#########

	instance_category_names = model_utils.instance_category_names(
		for_object=ooi,
		network_name=network_name
	)

	############
	# Tracking #
	############

	# initialize tracker
	init_tracker = tracker_utils.initialize_tracker(
		tracker=tracker,
		max_disappeared=max_disappeared,
		max_age=max_age,
		min_hits=min_hits,
		iou_threshold=iou_threshold
	)

	# load network
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	net = network_utils.load_network(
		looking_for_object=ooi,
		network_name=network_name,
		eval_mode=True,
		pre_trained_model=trained_model,
		device=device
	)

	# process the video stream
	video_stream_utils.process_video_stream(
		video_stream=video,
		network=net,
		device=device,
		instance_category_names=instance_category_names,
		looking_for_object=ooi,
		confidence_score=confidence_score,
		threshold_masks=threshold_masks,
		tracker=init_tracker,
		plot_detector_bbox=plot_detector_bbox,
		plot_tracker_bbox=plot_tracker_bbox,
		plot_keypoints=plot_keypoints,
		plot_labels=plot_labels,
		plot_pose=plot_pose,
		plot_mask=plot_mask,
		opacity=opacity,
		plot_id=plot_id,
		save_images=save_images,
		image_folder=image_folder,
		write_tracking_data=write_tracking_data,
		tracking_data_folder=tracking_data_folder,
		challenge=challenge,
		tracking_data_file_name=tracking_data_file_name,
		trained_model=trained_model,
		full_screen=full_screen
	)


if __name__ == '__main__':

	my_args = parse_args()
	print("args: {}".format(my_args))

	main(my_args)
