#!/bin/sh
echo "This shell script setups environment to use GPUs on Exacloud system."
# Make it runnable by "chmod u+x setup_env.sh"
# Run it by "./setup_env.sh"

# I want to use Pytorch 1.12 along with Python 3.9 and Cuda 11.3
# I installed pytorch (after loading CUDA modules) using the following command:
#   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# Other packages:
#   pip install pandas scikit-learn opencv-python matplotlib openpyxl

# cd IshikawaLab/ashkan/fcn
# source ~/.bashrc
# conda activate py39torch112

export SPACK_ROOT=/home/exacloud/software/spack
. $SPACK_ROOT/share/spack/setup-env.sh
. $(spack location -i lmod)/lmod/lmod/init/bash

module load cuda-11.3.1-gcc-8.3.1-zbpvyum cudnn-8.2.0.53-11.3-gcc-8.3.1-y2h5srj