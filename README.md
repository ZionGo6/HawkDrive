# HawkDrive

This is the repo for dataset and codes of our paper 'HawkDrive: A Transformer-driven Visual Perception System for Autonomous Driving in Night Scene'

## Dataset
Follow this [link](https://www.baidu.com) to download the dataset.

## Docker build
```
Build two images for camera driver(./theimagingsource_ros/docker/) and perception modules(./docker/) following the general docker commands and scripts.
```

## Run the pipeline
```
# Start the stereo camera capturing in one terminal.
cd theimagingsource_ros/scripts/
./up.sh

# In another terminal, in the container of perception modules, run the corresponding nodes.
# To start Enhancement Node
cd Low_light_Enhancement/
python3 enhance.py

# To start Resize Node
cd Resize_Node/
python3 resize.py

# To start Seg Node
cd Segmentation/
python3 Seg_ROS.py

# To start Depth Estimation Node
cd Depth/
python3 infer_ROS.py
python3 DPT_ROS.py
```


This project was built on [Isaac-ROS](https://github.com/NVIDIA-ISAAC-ROS), [SegFormer](https://github.com/NVlabs/SegFormer), [Unimatch](https://github.com/autonomousvision/unimatch), [SNR-aware Low Light Enhancement](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance), [DPT](https://github.com/isl-org/DPT). Thanks for the great open-sourced work! 