#!/bin/sh
echo "This shell script setups environment to use GPUs on Exacloud system."
# Make it runnable by "chmod u+x setup_env.sh"
# Run it by "./setup_env.sh"

# cd IshikawaLab/ashkan/oct-3d-ohsu
# source ~/.bashrc
# conda activate py38tf25

export SPACK_ROOT=/home/exacloud/software/spack
. $SPACK_ROOT/share/spack/setup-env.sh
. $(spack location -i lmod)/lmod/lmod/init/bash

module load cuda-11.2.0-gcc-8.3.1-zmax27o cudnn-8.1.1.33-11.2-gcc-8.3.1-2nhdj5d