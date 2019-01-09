#!/usr/bin/env bash
#
#python -m tensorflow.python.tools.import_pb_to_tensorboard \
#  --model_dir=/home/sixd-ailabs/Develop/DL/MobileDL/PocketFlow/nets_builder/models_uqtf_eval/model_original.pb \
#  --log_dir=./tmp

 /home/sixd-ailabs/Develop/DL/TF/test/tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco \
 --input_file=/home/sixd-ailabs/Develop/DL/MobileDL/PocketFlow/nets_builder/models_uqtf_eval/model_original.pb \
 --output_file=/home/sixd-ailabs/Develop/DL/MobileDL/PocketFlow/nets_builder/models_uqtf_eval/model_quant.tflite \
 --input_shapes=1,96,96,3 \
 --input_arrays=net_input \
 --output_arrays=net_output \
 --inference_type=QUANTIZED_UINT8 \
 --mean_values=115