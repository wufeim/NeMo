#!/bin/bash
#PBS -N TrainNemoClsOOD
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=15:gpus=3:nvidiaRTX3090,mem=10gb,walltime=24:00:00
#PBS -q default-cpu
#PBS -m a
#PBS -M jesslen@informatik.uni-freiburg.de
#PBS -j oe

source /home/jesslen/.bashrc
conda activate nemo
source /misc/software/cuda/add_environment_cuda11.1.sh
cd /home/jesslen/Documents/Github/NeMo
# Print start time 
start=$(date)
echo "Start time: $start"
python scripts/inference.py \
    --cate all \
    --config config/3d_aware_cls_nemo_cls.yaml \
    --save_dir exp/3d_aware_cls_nemo_cls
# Print end time
end=$(date)
echo "End time: $end" 
# Print elapsed time
# echo "Elapsed time: $((($(date -d "$end" +%s) - $(date -d "$start" +%s)) / 60)) minutes"
