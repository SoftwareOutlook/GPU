module load h5utils/1.13.1-intel-17.0.2
module load OpenMPI/2.1.2-GCC-6.4.0-2.28
module load FFTW/3.3.7-gompi-2018a
module load gcc/8.1.0
module load GPUmodules
module load cuda/9.0.176
module load libBoost/1.66.0-gcc-4.8.5

C_INCLUDE_PATH=${HOME}/GPU/include:${HOME}/opt/include:${C_INCLUDE_PATH}
CPLUS_INCLUDE_PATH=${HOME}/GPU/include:${HOME}/opt/include:${CPLUS_INCLUDE_PATH}

LD_LIBRARY_PATH=${HOME}/opt/lib:${LD_LIBRARY_PATH}
