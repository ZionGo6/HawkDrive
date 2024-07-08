# theimagingsource_ros

## Folder structure:
- **src** - a folder for source code or python scripts;
- **docker** - a folder for all docker-related files: Dockerfile, docer-compose files, env files, etc.;

- **docker/scripts/build.sh** - a script that build the docker image;
- **docker/scripts/up.sh** - a script that run a docker container from the built docker image;
- **docker/scripts/into.sh** - a script that run a docker container from the built docker image;

- **src/stereocam_publisher/config** - a folder for config files (.yaml for ROS2 node config and .json for camera properties setup) with parameters. This folder should by binded to the docker container, so you can apply files inside docker;
- **src/stereocam_publisher/launch** - a folder for launch files (.py), that start the ROS node and load parameters from configuration files described above. This folder should by binded to the docker container, so you can run files inside docker;


## Usage:
In Jetson Agx Xavier, to launch the nodes and perform the publisher: 
``` bash
cd home/user/workspace
git clone git@gitlab.com:sk-isrl/sdc-kia/stereo/theimagingsource_ros.git
cd theimagingsource_ros/docker/scripts
sudo ./up.sh
```
