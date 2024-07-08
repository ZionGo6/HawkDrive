#!/bin/bash

cd "$(dirname "$0")"
cd ../..
export PRODUCTION_PATH=$PWD
export ARCH=`uname -m`
# export NUM_THREADS=`nproc`

xhost +
docker-compose --env-file $PRODUCTION_PATH/docker/up.env \
    -f $PRODUCTION_PATH/docker/up.yml \
    up $@
xhost -
