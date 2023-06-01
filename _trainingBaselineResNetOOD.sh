#!/bin/bash
#PBS -N TrainResNetOOD
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=28:gpus=1:nvidiaMinCC75,mem=10gb,walltime=12:00:00
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
ALL_CATEGORIES=("aeroplane"  "bicycle"  "boat"  "bus"  "car"  "chair"  "diningtable"  "motorbike"  "sofa"  "train")
for CATEGORY in "${ALL_CATEGORIES[@]}"
do
    echo "Training for category $CATEGORY"
    python scripts/train.py \
        --cate $CATEGORY \
        --config config/pose_estimation_3d_oodcv_resnet50_general.yaml \
        --save_dir exp/pose_estimation_3d_oodcv_resnet50_general_$CATEGORY
done
# Print end time
end=$(date)
echo "End time: $end" 
# Print elapsed time
echo "Elapsed time: $((($(date -d "$end" +%s) - $(date -d "$start" +%s)) / 60)) minutes"