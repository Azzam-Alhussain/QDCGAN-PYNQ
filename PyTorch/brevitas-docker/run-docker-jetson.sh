#!/bin/bash

set -e

DOCKER_GID=$(id -g)
DOCKER_GNAME=$(id -gn)
DOCKER_UNAME=$(id -un)
DOCKER_UID=$(id -u)
DOCKER_TAG="brevitas"
DOCKER_PASSWD="brevitas"
DOCKER_INST_NAME="$DOCKER_UNAME-brevitas-inst"

PARENT_PATH=$(readlink -f ../../)

docker build \
    -t ${DOCKER_TAG} \
    -f Dockerfile_jetson \
    --build-arg GID=$DOCKER_GID \
    --build-arg GNAME=$DOCKER_GNAME \
    --build-arg UNAME=$DOCKER_UNAME \
    --build-arg UID=$DOCKER_UID \
    --build-arg PASSWD=$DOCKER_PASSWD \
    .

DOCKER_EXEC="docker run --rm --runtime nvidia --network host "
DOCKER_EXEC+="--name $DOCKER_INST_NAME -i -t "
DOCKER_EXEC+="-v $PARENT_PATH:/home/$DOCKER_UNAME/QDCGAN "
DOCKER_EXEC+="${DOCKER_TAG} /bin/bash"

$DOCKER_EXEC

