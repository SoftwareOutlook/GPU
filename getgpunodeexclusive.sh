#!/bin/bash

srun --pty -p gpu-exclusive --gres=gpu:4 /bin/bash

