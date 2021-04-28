#!/bin/bash

###Snippet from http://stackoverflow.com/questions/59895/
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
###end snippet

docker build -t luntlab/cs547_project:latest --target project_base ${SCRIPT_DIR}

#docker push luntlab/cs547_project:latest
