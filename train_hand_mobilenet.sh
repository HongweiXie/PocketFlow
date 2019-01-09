#!/usr/bin/env bash
cd nets_builder
python mobilent_at_hand_run --model_http_url https://api.ai.tencent.com/pocketflow --data_dir_local=/home/sixd-ailabs/Develop/Human/Hand/hand_dataset/hand-classification/tfrecord --mobilenet_version=1 --mobilenet_depth_mult=0.25 --batch_size=128 --checkpoint_path=/home/sixd-ailabs/Downloads/mobilenet_v1_0.25_128 --checkpoint_exclude_scopes=model/MobilenetV1/Logits --learner=uniform-tf

python mobilent_at_hand_run --model_http_url https://api.ai.tencent.com/pocketflow --data_dir_local=/home/sixd-ailabs/Develop/Human/Hand/hand_dataset/hand-classification/tfrecord --mobilenet_version=2 --mobilenet_depth_mult=0.35 --batch_size=128 --checkpoint_path=/home/sixd-ailabs/Downloads/mobilenet_v1_0.25_128 --checkpoint_exclude_scopes=model/MobilenetV1/Logits