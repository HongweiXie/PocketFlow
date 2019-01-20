#!/usr/bin/env bash

./scripts/run_local_test.sh nets_builder/mobilenet_at_places_run.py -n=1 \
    --mobilenet_version=2 \
    --mobilenet_depth_mult=1.4 \
    --batch_size=8 \
    --lrn_rate_init=0.01 \
    --checkpoint_path=/home/sixd-ailabs/Downloads/mobilenet_v2_1.4_224 \
    --checkpoint_exclude_scopes=model/MobilenetV2/Logits \
    --nb_classes=7 \
    --nb_smpls_train=38900 \
    --nb_smpls_val=3800 \
    --nb_smpls_eval=3800
