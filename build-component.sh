#!/usr/bin/env bash

#globals
COMPONENT_NAME=$1
TAG_TEMP=$2
TAG=${TAG_TEMP:=latest}
TEMP=$3
PUSH=${TEMP:=-NO}
IMAGE=prod-nexus.sprinklr.com:8123/intuition/${COMPONENT_NAME}:${TAG}



function usage() {
  echo "./build-component.sh <component> <docker-tag> <push?>"
  echo "component = one among [agent-builder-api]"
  echo "docker-tag = tag for the image (default latest)"
  echo "push? = YES or NO (push the image to repo)
                default:NO
                "
  exit 1
}

function ping(){
    echo "Building from docker file path::"
}
function check_component() {
   echo "COMPONENT_NAME: " ${COMPONENT_NAME}
  case ${COMPONENT_NAME} in
    agent-builder-api)
    DOCKERFILE_PATH=./docker/api/Dockerfile
    ;;
    *)
      echo "Invalid component" && exit 1 ;;
  esac
  echo "<Building Image>
        dockerfile path ::" ${DOCKERFILE_PATH} "
        tag             ::" ${TAG}
}


function build(){
    if [ ! -z "${PROD_GITLAB_INT_IP}" ]; then
        docker build --add-host prod-gitlab.sprinklr.com:${PROD_GITLAB_INT_IP} --add-host prod-nexus.sprinklr.com:${PROD_NEXUS_INT_IP} --add-host nqa-nexus.sprinklr.com:${PROD_NEXUS_INT_IP} -f ${DOCKERFILE_PATH} -t ${IMAGE} .
    else
        docker build -f ${DOCKERFILE_PATH} -t ${IMAGE} .
    fi
    case ${PUSH} in
    YES)
    echo "Pushing Image:: " ${IMAGE}
    docker push ${IMAGE}
    ;;
    NO)
    ;;
    *)
    echo "Invalid component" && exit 1 ;;
    esac
}

if [ "$#" < 3 ]; then
  usage
fi

check_component
build


