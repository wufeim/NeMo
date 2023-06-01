#!/bin/bash
#PBS -N InferenceResNetOOD
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=28:gpus=1:nvidiaRTX2080Ti,mem=10gb,walltime=24:00:00
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
    echo "Inference for category $CATEGORY"
    python scripts/inference.py \
        --cate $CATEGORY \
        --config config/pose_estimation_3d_oodcv_resnet50_general.yaml \
        --save_dir exp/pose_estimation_3d_oodcv_resnet50_general_$CATEGORY \
        --checkpoint exp/pose_estimation_3d_oodcv_resnet50_general_$CATEGORY/ckpts/model_200.pth
done

# Print end time
end=$(date)
echo "End time: $end" 
# Print elapsed time
echo "Elapsed time: $((($(date -d "$end" +%s) - $(date -d "$start" +%s)) / 60)) minutes"