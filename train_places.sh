#!/usr/bin/env bash

./scripts/run_local_test.sh nets/resnet_at_places_run.py -n=8 \
    --resnet_size=50 \
    --batch_size=32 \
    --checkpoint_path=/home/hongwei.xhw/Develop/TF/model_zoon/pocketflow_reset_50 \
    --checkpoint_exclude_scopes=model/resnet_model/dense
