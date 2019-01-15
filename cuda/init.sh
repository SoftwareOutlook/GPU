module load GPUmodules
module load cuda/9.0.176
module load libBoost/1.66.0-gcc-4.8.5

C_INCLUDE_PATH=${HOME}/GPU/include:${C_INCLUDE_PATH}
CPLUS_INCLUDE_PATH=${HOME}/GPU/include:${CPLUS_INCLUDE_PATH}
