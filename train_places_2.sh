#!/usr/bin/env bash

./scripts/run_local_test.sh nets/resnet_at_places_run.py -n=8 \
    --resnet_size=50 \
    --batch_size=32 \
    --lrn_rate_init=0.01 \
    --checkpoint_path=./models \
    --checkpoint_exclude_scopes=model/resnet_model/dense
