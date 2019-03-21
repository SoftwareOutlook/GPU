module load gcc
module load cuda openmpi
module load fftw

OPT=${HOME}/opt
INCLUDE_PATH=${OPT}/include:${INCLUDE_PATH}
LD_LIBRARY_PATH=${OPT}/lib:${LD_LIBRARY_PATH}
PATH=${OPT}/bin:${PATH}

INCLUDE_PATH=${HOME}/GPU/include:${INCLUDE_PATH}
