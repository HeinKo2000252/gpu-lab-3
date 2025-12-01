#!/bin/bash
#SBATCH -J lab3_gpu
#SBATCH -p gpuserv
#SBATCH -e lab3.%j.err
#SBATCH -o lab3.%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:10:00

module load nvidia/cuda

echo "Current path=`pwd`"
echo "node=`hostname`"
echo "nproc=`nproc`"
echo $SLURM_JOBID
echo $SLURM_SUBMIT_DIR
echo $SLURM_JOB_NODELIST
echo $SLURM_CPUS_PER_TASK
echo $SLURM_NTASKS

cd /home/mag/s_khomchenko/lab3

echo "=== Checking GPU availability ==="
nvidia-smi

echo "=== Compiling lab3.cu ==="
nvcc lab3.cu -o img_filter -lpng

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "=== Running lab3 program ==="
    ./img_filter test_img.png out_blur.png blur
else
    echo "Compilation failed!"
    exit 1
fi

echo "Job completed successfully!"