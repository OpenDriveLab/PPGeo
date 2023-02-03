import math
import json
import random
random.seed(0)
import numpy as np
from tqdm import tqdm
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from scipy.spatial.transform import Rotation
from numpyencoder import NumpyEncoder

# Hyper-Params
DATA_ROOT = 'data/nuscenes'
SPLIT = 'v1.0-trainval'
NUM_RGB_IMGS = 1
NUM_FUTURE_RGB_IMGS = 2
NUM_FUTURE_TRAJECTORY_PTS = 6
OUTPUT_JSON_NAME = 'data/p3_6pts_can_bus_%s.json'
GET_CAN_BUS = True

TEMPORAL = False


sensors_tree = {
    'vehicle_monitor':
    [
        'brake',
        'steering',
        'throttle',
    ],
}

def find_nearest_index(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def get_samples(nusc, scenes, nusc_can=None):
    samples = []
    # list of dicts, where
    # 'img': LIST of filenames 0, 1, ..., NUM_RGB_IMGS - 1. 
    # NUM_RGB_IMGS - 1 is the frame of 'current' timestamp
    # 'pt_%d': LIST of future points offset by current img 0, 1, ..., NUM_FUTURE_TRAJECTORY_PTS - 1
    # 0 is the point of the 'very next' timestamp
    for scene in tqdm(scenes, ncols=0):
        assert len(scene) >= NUM_RGB_IMGS + NUM_FUTURE_TRAJECTORY_PTS
        valid_start_tokens = scene[NUM_RGB_IMGS-1 : -NUM_FUTURE_TRAJECTORY_PTS]
        if TEMPORAL:
            cur_scene_samples = []
        # CAN BUS
        if nusc_can is not None:
            can_bus_cache = dict()
            scene_token = nusc.get('sample', valid_start_tokens[0])['scene_token']
            scene_name = nusc.get('scene', scene_token)['name']

            has_can_bus_data = True
            for message_name, keys in sensors_tree.items():
                try:
                    can_data = nusc_can.get_messages(scene_name, message_name)
                except Exception:
                    has_can_bus_data = False
                    continue
                can_bus_cache['%s.utime' % (message_name)] = np.array([m['utime'] for m in can_data])
                if len(can_bus_cache['%s.utime' % message_name]) == 0:
                    has_can_bus_data = False
                    continue
                for key_name in keys:
                    can_bus_cache['%s.%s' % (message_name, key_name)] = np.array([m[key_name] for m in can_data])

            if not has_can_bus_data:
                print('Error: %s does not have any CAN bus data!' % scene_name)
                continue

        for idx, cur_token in enumerate(valid_start_tokens):
            img_tokens = scene[idx:idx+NUM_RGB_IMGS+NUM_FUTURE_RGB_IMGS]
            point_tokens = scene[idx+NUM_RGB_IMGS:idx+NUM_RGB_IMGS+NUM_FUTURE_TRAJECTORY_PTS]
            
            cam_front_data = nusc.get('sample_data', nusc.get('sample', cur_token)['data']['CAM_FRONT'])
            # Images
            imgs = list(nusc.get('sample_data', nusc.get('sample', token)['data']['CAM_FRONT'])['filename'] for token in img_tokens)
            
            # Ego poses
            cur_ego_pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
            ego_rotation_matrix = Rotation.from_quat(np.array(cur_ego_pose['rotation'])[[1,2,3,0]]).as_matrix()
            ego_tranlation = np.array(cur_ego_pose['translation'])
            ego_rotation_matrix_inv = np.linalg.inv(ego_rotation_matrix)
            ego_tranlation_inv = -ego_tranlation

            future_poses = list(nusc.get('ego_pose', nusc.get('sample_data', nusc.get('sample', token)['data']['CAM_FRONT'])['ego_pose_token'])['translation'] for token in point_tokens)
            future_poses = list(ego_rotation_matrix_inv @ (np.array(future_pose)+ego_tranlation_inv) for future_pose in future_poses)
            future_poses = list(list(p) for p in future_poses)  # for json

            # Camera Matrices
            calibration_para = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
            camera_intrinsic = np.array(calibration_para['camera_intrinsic'])
            camera_rotation_matrix = Rotation.from_quat(np.array(calibration_para['rotation'])[[1,2,3,0]]).as_matrix()
            camera_translation = np.array(calibration_para['translation'])
            camera_rotation_matrix_inv = np.linalg.inv(camera_rotation_matrix)
            camera_translation_inv = -camera_translation
            camera_extrinsic = np.vstack((np.hstack((camera_rotation_matrix_inv, camera_translation_inv.reshape((3, 1)))), np.array([0, 0, 0, 1])))

            cur_sample_to_append = dict(
                imgs=imgs,
                future_poses=future_poses,
                camera_intrinsic=camera_intrinsic.tolist(),
                camera_extrinsic=camera_extrinsic.tolist(),
                camera_translation_inv=camera_translation_inv.tolist(),
                camera_rotation_matrix_inv=camera_rotation_matrix_inv.tolist(),
            )

            # CAN BUS
            if nusc_can is not None:
                img_timestamp = nusc.get('sample_data', nusc.get('sample', img_tokens[-1])['data']['CAM_FRONT'])['timestamp']
                cur_sample_to_append['img_utime'] = img_timestamp
                for message_name, keys in sensors_tree.items():
                    message_utimes = can_bus_cache['%s.utime' % message_name]
                    nearest_index = find_nearest_index(message_utimes, img_timestamp)
                    can_bus_time_delta = abs(message_utimes[nearest_index] - img_timestamp)  # ideally should be less than half the sample rate (2Hz * 2 = 4Hz)
                    if can_bus_time_delta >= 0.25 * 1e6:
                        print('Warning', scene_name, message_utimes[nearest_index], img_timestamp, can_bus_time_delta)
                    cur_sample_to_append['can_bus.%s.utime' % message_name] = message_utimes[nearest_index]
                    cur_sample_to_append['can_bus.%s.can_bus_delta' % message_name] = can_bus_time_delta
                    for key_name in keys:
                        can_bus_value = can_bus_cache['%s.%s' % (message_name, key_name)][nearest_index]
                        if isinstance(can_bus_value, np.ndarray):
                            can_bus_value = can_bus_value.tolist()
                        cur_sample_to_append['can_bus.%s.%s' % (message_name, key_name)] = can_bus_value
            if TEMPORAL:
                cur_scene_samples.append(cur_sample_to_append)
            else:
                samples.append(cur_sample_to_append)

        if TEMPORAL:
            samples.append(cur_scene_samples)

    return samples


# Load NuScenes dataset
nusc = NuScenes(version=SPLIT, dataroot=DATA_ROOT, verbose=True)
nusc_can = NuScenesCanBus(dataroot=DATA_ROOT) if GET_CAN_BUS else None

def get_scenes(split, nusc,  nusc_can):
    
    blacklist = [419] + nusc_can.can_blacklist  # # scene-0419 does not have vehicle monitor data
    blacklist = ['scene-' + str(scene_no).zfill(4) for scene_no in blacklist]

    scenes = create_splits_scenes()[split][:]
    for scene_no in blacklist:
        if scene_no in scenes:
            scenes.remove(scene_no)

    scene_name_dict = {}
    for scene in nusc.scene:
        scene_name_dict[scene['name']] = scene


    all_scenes = []
    for scene_no in scenes:
        scene = scene_name_dict[scene_no]
        cur_token = scene['first_sample_token']
        cur_scene_tokens = []
        while cur_token != '':
            cur_scene_tokens.append(cur_token)
            cur_sample = nusc.get('sample', cur_token)
            cur_token = cur_sample['next']

        all_scenes.append(cur_scene_tokens)

    return all_scenes

train_scenes = get_scenes('train', nusc, nusc_can)
val_scenes = get_scenes('val', nusc, nusc_can)

train_samples = get_samples(nusc, train_scenes, nusc_can)
val_samples = get_samples(nusc, val_scenes, nusc_can)

print(len(train_samples), len(val_samples))

json.dump(train_samples, open(OUTPUT_JSON_NAME % 'train', 'w'), indent='\t', cls=NumpyEncoder)
json.dump(val_samples, open(OUTPUT_JSON_NAME % 'val', 'w'), indent='\t', cls=NumpyEncoder)
