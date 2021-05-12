#!/bin/bash

###Snippet from http://stackoverflow.com/questions/59895/
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
###end snippet

CONTAINER_VERSION="v0.2"

docker build -t luntlab/cs547_project:${CONTAINER_VERSION} --target project_base ${SCRIPT_DIR} || exit 1
docker tag luntlab/cs547_project:${CONTAINER_VERSION} luntlab/cs547_project:latest

docker push luntlab/cs547_project:${CONTAINER_VERSION}
docker push luntlab/cs547_project:latest
