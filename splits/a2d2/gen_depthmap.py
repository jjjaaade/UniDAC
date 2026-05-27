import os
import os.path as osp
import json
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from p_tqdm import p_map
import argparse



def get_lidar_filename(img_json):
    with open(img_json, "r") as f:
        info_dict = json.load(f)
    
    return info_dict["pcld_npz"]

def func(info):
    
    img_dir = info["img_dir"]
    lidar_dir = info["lidar_dir"]
    depth_dir = info["depth_dir"]
    frame = info["frame"]

    img_path = osp.join(img_dir, frame)
    img_json = img_path.replace(".png", ".json")
    lidar_frame = get_lidar_filename(img_json)
    lidar_path = osp.join(lidar_dir, lidar_frame)
    depth_path = osp.join(depth_dir, frame.replace("camera", "depth"))
    
    img = np.array(Image.open(img_path))
    
    lidar_front_center = np.load(lidar_path)

    rows = lidar_front_center["pcloud_attr.row"]
    cols = lidar_front_center["pcloud_attr.col"]
    depth = lidar_front_center["pcloud_attr.depth"]
    
    depth_arr = np.zeros(img.shape[:2], dtype=np.float32)

    depth_arr[rows.astype(int), cols.astype(int)] = depth

    depth_im = Image.fromarray((depth_arr*256).astype(np.uint16))
    depth_im.save(depth_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the A2D2 dataset")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    cam_lidar_dir = osp.join(DATA_DIR, "camera_lidar")
    scene_list = [scene for scene in os.listdir(cam_lidar_dir) if os.path.isdir(osp.join(cam_lidar_dir, scene))]

    for scene in scene_list:
        print("Scene: ", scene)
        data_info = []
        img_dir = osp.join(cam_lidar_dir, scene, "camera/cam_front_center")
        lidar_dir = osp.join(cam_lidar_dir, scene, "lidar/cam_front_center")
        depth_dir = osp.join(cam_lidar_dir, scene, "depth/cam_front_center")
        os.makedirs(depth_dir, exist_ok=True)

        for frame in os.listdir(img_dir):
            if ".png" in frame:
                data_info.append({"img_dir": img_dir,
                                  "lidar_dir": lidar_dir,
                                  "depth_dir": depth_dir,
                                  "frame": frame})
                
        r = p_map(func, data_info, num_cpus=os.cpu_count())    