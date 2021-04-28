#!/bin/sh

###Snippet from http://stackoverflow.com/questions/59895/
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
###end snippet

IMAGE_NAME="luntlab/cs547_project:latest"


docker run -it --rm -v${SCRIPT_DIR}:/workspace/project --ipc=host ${IMAGE_NAME}
