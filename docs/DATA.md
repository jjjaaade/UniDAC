## Dataset Preparation Guidelines

<!-- ![data](table_data_coverage.png) -->
Our training set builds on top of the training set of [Depth Any Camera](https://github.com/yuliangguo/depth_any_camera/tree/main).
Follow DAC's [data preparation](https://github.com/yuliangguo/depth_any_camera/blob/main/docs/DATA.md) guide to prepare **HM3D**, **HyperSim**, **Taskonomy**, **DDAD**, **Lyft**, **KITTI** and **NYU** datasets and ensure they are in the required data structure format.

We provide below the data preparation steps, i.e., **Argoverse2**, **A2D2**, **iBims** and **NuScenes**. The required data format for these datasets are as follows:

```bash
depth_any_camera
├── datasets
│      ├── hm3d
│      ├── hypersim
│      ├── taskonomy
│      ├── scannetpp
│      ├── gibson_v2
│      ├── matterport3d
│      ├── ddad
│      ├── lyft
│      ├── kitti360
│      ├── kitti
│      ├── nyu
│      ├── a2d2
│      │     ├── camera_lidar
│      │     │     ├── 20180810_150607
│      │     │     │     ├── camera
│      │     │     │     ├── depth
│      │     │     ├── 20190401_121727
│      │     │     ├── ...
│      ├── argoverse2
│      │     ├── train
│      │     │     ├── 5d391e54-adec-3584-adf0-5025d7564e1b
│      │     │     │     ├── calibration
│      │     │     │     ├── sensors
│      │     │     │     │     ├──cameras
│      │     │     │     │     │     ├──ring_*
│      │     │     │     │     ├──lidar
│      │     │     ├── ...
│      │     ├── depth
│      │     │     ├── 5d391e54-adec-3584-adf0-5025d7564e1b
│      │     │     │     ├──ring_*
│      │     │     ├── ...
│      │     ├── ...
│      ├── ibims
│      │     ├── m1455541
│      │     │     ├── ibims1_core_raw
│      │     │     │     ├── rgb
│      │     │     │     ├── depth
│      │     │     │     ├── ...
│      ├── nuscenes
│      │     ├── samples
│      │     ├── depth
│      │     ├── point_info
│      │     ├── rgb
│      │     ├── ...
```

Ensure that the datasets are soft-linked to folders under the `datasets` directory and follow the data structure mentioned [here](https://github.com/yuliangguo/depth_any_camera/blob/main/docs/DATA.md), so that our config files and scripts can be directly utilized.

## Testing Datasets


### **iBims-1**
Download the official dataset from [here](https://www.asg.ed.tum.de/lmf/ibims1/) which includes RGB-D data samples.
The dataset is soft-linked in `datasets/ibims` and splits are saved [here](../splits/kitti)

### **NuScenes**
Download the official dataset from [here](https://www.nuscenes.org/nuscenes), which includes camera, lidar and calibration data for 1000 scenes and install the devkit from the [offical NuScenes repository](http://github.com/nutonomy/nuscenes-devkit).


Run the following code to generate depthmaps from LiDAR. Download masks from [here](https://github.com/ShngJZ/RePLAy-Release) to remove projective artifacts while generating depthmaps from LiDAR.

```bash
cd ./UniDAC
python ./splits/nuscenes/gen_depthmap.py --data_dir ./dataset/nuscenes --split val
```

The dataset is soft-linked to `datasets/nuscenes`, and splits are saved [here](../splits/nuscenes).


## Training Datasets

### Argoverse2
Download the dataset from [here](https://www.argoverse.org/av2.html#download-link) which includes camera, LiDAR, and calibration data.
Refer to their official github repository, [av2-api](https://github.com/argoverse/av2-api/blob/b7321d1f71f6ce0ecdd151f4f2b648338c191edd/src/av2/datasets/sensor/av2_sensor_dataloader.py#L415), to generate the depth maps.
Ensure the generated depthmaps follow the file structure as mentioned above.

The dataset should be soft-linked to `datasets`. We randomly select 500K samples and provide the corresponding split file [here](../splits/argoverse2/argoverse2_train.txt).

### A2D2
Download the camera, LiDAR and calibration data from [here](https://www.a2d2.audi/en/download/) for all the scenes, namely, Gaimersheim, Ingolstadt, and Munich.

The dataset should be soft-linked to `datasets`. We ignore the front-camera samples as mentioned in the paper and provide the generated split file
[here](../splits/a2d2/a2d2_train.txt).

Run the following script to generate the depthmaps

```
python splits/a2d2/gen_depthmap.py --data_dir ./datasets/a2d2
```
---
