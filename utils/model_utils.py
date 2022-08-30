import sys
import os
sys.path.insert(0, os.path.abspath('./models'))
import pigeon_model
import coco_model
import mouse_model
import cowbird_model

print("[INFO] initializing object of interest...")


def is_coco_instance(for_object, network_name):

    if for_object == 'pigeon':
        var_is_coco_instance = pigeon_model.IS_COCO_INSTANCE
    elif for_object == 'cowbird':
        var_is_coco_instance = cowbird_model.IS_COCO_INSTANCE
    elif for_object in coco_model.COCO_INSTANCE_CATEGORY_NAMES:
        if for_object == 'mouse':
            if network_name == 'KeypointRCNN':
                var_is_coco_instance = mouse_model.IS_COCO_INSTANCE
            else:
                var_is_coco_instance = coco_model.IS_COCO_INSTANCE
        else:
            var_is_coco_instance = coco_model.IS_COCO_INSTANCE
    else:
        print('[ERROR] !!! select available object of interest !!!')
        assert False

    return var_is_coco_instance


def instance_category_names(for_object, network_name):

    if for_object == 'pigeon':
        var_instance_category_names = pigeon_model.SELF_DEFINED_INSTANCE_CATEGORY_NAMES
    elif for_object == 'cowbird':
        var_instance_category_names = cowbird_model.SELF_DEFINED_INSTANCE_CATEGORY_NAMES
    elif for_object in coco_model.COCO_INSTANCE_CATEGORY_NAMES:
        if for_object == 'mouse':
            if network_name == 'KeypointRCNN':
                var_instance_category_names = mouse_model.SELF_DEFINED_INSTANCE_CATEGORY_NAMES
            else:
                var_instance_category_names = coco_model.COCO_INSTANCE_CATEGORY_NAMES
        else:
            var_instance_category_names = coco_model.COCO_INSTANCE_CATEGORY_NAMES
    else:
        print('[ERROR] !!! select available object of interest !!!')
        assert False

    return var_instance_category_names


def keypoint_names(for_object):

    if for_object == 'pigeon':
        var_keypoint_names = pigeon_model.PIGEON_KEYPOINT_NAMES
    elif for_object == 'cowbird':
        var_keypoint_names = cowbird_model.COWBIRD_KEYPOINT_NAMES
    elif for_object == 'person':
        var_keypoint_names = coco_model.COCO_PERSON_KEYPOINT_NAMES
    elif for_object == 'mouse':
        var_keypoint_names = mouse_model.DLC_MOUSE_KEYPOINT_NAMES
    else:
        print('[ERROR] !!! select available object of interest !!!')
        assert False

    return var_keypoint_names
