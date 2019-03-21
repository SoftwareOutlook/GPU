export C_INCLUDE_PATH=${HOME}/GPU/include:/apps/packages/cuda/10.0/include:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=${HOME}/GPU/include:/apps/packages/cuda/10.0/include:${CPLUS_INCLUDE_PATH}


export OPT=${HOME}/opt
export C_INCLUDE_PATH=${OPT}/include:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=${OPT}/include:${CPLUS_INCLUDE_PATH}
export LD_LIBRARY_PATH=${OPT}/lib:${LD_LIBRARY_PATH}

module load gcc
module load cuda
module load openmpi
