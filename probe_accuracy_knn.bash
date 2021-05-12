#!/bin/bash

#TODO: Make commandline parameter
#PROBE_RUN="20mi8lwz"
#PROBE_RUN_MODEL="resnet18 --batch_size 300"

#
#PROBE_RUN="20mi8lwz"
#PROBE_RUN_MODEL="resnet18 --batch_size 200"

#PROBE_RUN="t1wlcqry"
#PROBE_RUN_MODEL="DeepRank --batch_size 20"

PROBE_RUN="2vvezipo"
PROBE_RUN_MODEL="rescaled_resnet18 --batch_size 20"

wandb pull -e uiuc-cs547-2021sp-group36 -p image_similarity ${PROBE_RUN} || exit 1

mv model_state.pt ${PROBE_RUN}_model_state.pt

python src/eval_neighbors.py --model ${PROBE_RUN_MODEL} --weight_file ${PROBE_RUN}_model_state.pt || exit 1
