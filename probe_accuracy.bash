#!/bin/bash

#TODO: Make commandline parameter
PROBE_RUN="2vvezipo"
PROBE_RUN_MODEL="rescaled_resnet18"

#PROBE_RUN="t1wlcqry"
#PROBE_RUN_MODEL="DeepRank"

wandb pull -e uiuc-cs547-2021sp-group36 -p image_similarity ${PROBE_RUN} || exit 1

mv model_state.pt ${PROBE_RUN}_model_state.pt

python src/inference_main.py --model ${PROBE_RUN_MODEL} --weight_file ${PROBE_RUN}_model_state.pt --out /tmp/${PROBE_RUN}_emb.txt --best_matches ${PROBE_RUN}_matches.txt || exit 1

python << EOF
import numpy as np

fn = "${PROBE_RUN}_matches.txt"

with open(fn,"r") as infile:
    header = infile.readline()
    #print(header)
    data = np.loadtxt(infile,dtype=("S"))
    query_names = data[:,0]
    query_labels = data[:,1]
    filenames = data[:,2::2]
    filelabels = data[:,3::2]

test_accuracy = (query_labels == filelabels[:,0]).mean()

print("Test Accuracy {}".format(test_accuracy))
EOF
