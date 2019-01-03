#!/usr/bin/env bash

python nets/resnet_at_places_run.py \
    --model_http_url https://api.ai.tencent.com/pocketflow \
    --data_dir_local /home/sixd-ailabs/Develop/Scene/Places2/places365_standard/tfrecord \
    --resnet_size=50 \
    --batch_size=8 \
    --save_path=/home/sixd-ailabs/Develop/DL/MobileDL/PocketFlow/models \
    --checkpoint_path=/home/sixd-ailabs/Downloads/models \
    --checkpoint_exclude_scopes=model/resnet_model/dense