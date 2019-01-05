#!/bin/bash

# default arguments
nb_gpus=1

# parse arguments passed from the command line
py_script="$1"
shift
for i in "$@"
do
  case "$i" in
    -n=*|--nb_gpus=*)
    nb_gpus="${i#*=}"
    shift
    ;;
    *)
    # unknown option
    ;;
  esac
done
extra_args=`python utils/get_path_args.py local ${py_script} path.conf`
extra_args="$@ ${extra_args}"
echo "Python script: ${py_script}"
echo "# of GPUs: ${nb_gpus}"
echo "extra arguments: ${extra_args}"

# obtain list of idle GPUs
idle_gpus=`python utils/get_idle_gpus.py ${nb_gpus}`
export CUDA_VISIBLE_DEVICES=${idle_gpus}

# re-create the logging directory
rm -rf logs && mkdir logs

# execute the specified Python script with one or more GPUs
cp -v ${py_script} main.py
if [ ${nb_gpus} -eq 1 ]; then
  echo "multi-GPU training disabled"
  python main.py ${extra_args}
elif [ ${nb_gpus} -le 8 ]; then
  echo "multi-GPU training enabled"
  options="-np ${nb_gpus} -bind-to none -map-by slot"
  mpirun ${options} python main.py --enbl_multi_gpu ${extra_args}
fi
