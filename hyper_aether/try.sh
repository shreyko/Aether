#!/bin/bash
#SBATCH --job-name=neurframeq_download
#SBATCH --output=job_logs/neurframeq_download%j.log
#SBATCH --nodes=1
#SBATCH --partition=spgpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=jjparkcv_owned1

source ~/.bashrc
module load gcc/11
module load cuda/12.1
source /gpfs/accounts/jjparkcv_root/jjparkcv0/gpranav/sca/Aether/hyper_aether/.venv/bin/activate

cd /gpfs/accounts/jjparkcv_root/jjparkcv0/gpranav/sca/Aether/hyper_aether/

mkdir -p job_logs

export HYPER_AETHER_BACKEND=vllm
export HYPER_AETHER_MODEL="${HYPER_AETHER_MODEL:-Qwen/Qwen3.5-9B}"

python main.py