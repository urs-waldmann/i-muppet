import sys
import os
sys.path.insert(0, os.path.abspath('./sort'))
from sort import Sort


def initialize_tracker(tracker, max_disappeared=50, max_age=1, min_hits=3, iou_threshold=0.3):

    print("[INFO] initializing tracker...")

    # Centroid Tracker
    if tracker == 'ct':
        initialized_tracker = CentroidTracker(
            max_disappeared=max_disappeared  # default: max_disappeared=50
        )
    # SORT
    elif tracker == 'sort':
        initialized_tracker = Sort(
            max_age=max_age,  # default: max_age=1
            min_hits=min_hits,  # default: min_hits=3
            iou_threshold=iou_threshold  # default: iou_threshold=0.3
        )
    else:
        print('[ERROR] !!! select available tracker !!!')
        assert False

    return initialized_tracker
