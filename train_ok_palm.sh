#!/usr/bin/env bash


python nets_builder/mobilenet_at_hand_run.py \
--data_dir_local=/home/sixd-ailabs/Downloads/cropokpalm/tfrecord_72 \
--mobilenet_version=1 \
--mobilenet_depth_mult=0.5 \
--batch_size=32 \
--checkpoint_path=/home/sixd-ailabs/Downloads/mobilenet_v1_0.5_128 \
--checkpoint_exclude_scopes=model/MobilenetV1/Logits \
--learner=full-prec \
--nb_epochs_rat=1.0 \
--nb_smpls_train=16000 \
--nb_smpls_eval=1800 \
--nb_smpls_val=1800 \
--nb_classes=5

python nets_builder/mobilenet_at_hand_run.py \
--data_dir_local=/home/sixd-ailabs/Downloads/cropokpalm/tfrecord_72 \
--mobilenet_version=1 \
--mobilenet_depth_mult=0.5 \
--batch_size=16 \
--learner=uniform-tf \
--nb_epochs_rat=0.3 \
--nb_smpls_train=16000 \
--nb_smpls_eval=1800 \
--nb_smpls_val=1800 \
--nb_classes=5