module load h5utils/1.13.1-intel-17.0.2
module load OpenMPI/4.0.0-GCC-8.2.0-2.31.1
module load gcc/8.1.0
module load GPUmodules
module load cuda/9.0.176
module load libBoost/1.66.0-gcc-4.8.5

C_INCLUDE_PATH=${HOME}/GPU/include:${C_INCLUDE_PATH}
CPLUS_INCLUDE_PATH=${HOME}/GPU/include:${CPLUS_INCLUDE_PATH}
