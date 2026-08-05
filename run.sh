#!/bin/bash
#SBATCH --job-name=mnist_model
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=you@email.com
#SBATCH --time=2:00:00

#SBATCH --output=logs/mnist_model_%j.log
#SBATCH --error=logs/mnist_model_%j.err

#SBATCH --nodelist=hpc-novel-gpu[01-06]
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Enroot config
#SBATCH --container-image=/home/shared/air/enroot-images/pytorch_n_friends.sqsh
#SBATCH --container-mount-home
#SBATCH --container-mounts=/home/shared:/home/shared:ro

# Unload modules that may interfere
module purge

# Go to working directory
cd "$SLURM_SUBMIT_DIR"

# Run training script
echo "Startng job: $PWD"
srun bash -c "cd $SLURM_SUBMIT_DIR && source .venv/bin/activate && python mnist.py"
