#!/usr/bin/env bash

python nets_builder/mobilenet_at_hand_run.py --model_http_url https://api.ai.tencent.com/pocketflow --data_dir_local=/home/sixd-ailabs/Develop/Human/Hand/hand_dataset/hand-classification2/tfrecord_72 --mobilenet_version=1 --mobilenet_depth_mult=0.25 --batch_size=128 --checkpoint_path=/home/sixd-ailabs/Downloads/mobilenet_v1_0.25_128 --checkpoint_exclude_scopes=model/MobilenetV1/Logits

python nets_builder/mobilenet_at_hand_run.py --model_http_url https://api.ai.tencent.com/pocketflow --data_dir_local=/home/sixd-ailabs/Develop/Human/Hand/hand_dataset/hand-classification2/tfrecord_72 --mobilenet_version=1 --mobilenet_depth_mult=0.25 --batch_size=128 --checkpoint_path=/home/sixd-ailabs/Downloads/mobilenet_v1_0.25_128 --checkpoint_exclude_scopes=model/MobilenetV1/Logits --learner=uniform-tf --nb_epochs_rat=0.3

