#!/bin/bash

singularity_path=/scratch/satrajit.chakrabarty/singularities/mrcnn3d

cd ~/mrcnn3d_mdt/cuda_functions/nms_2D/src/cuda
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_50
cd ../../
singularity exec --nv -B /scratch:/scratch $singularity_path python3 build.py


cd ~/mrcnn3d_mdt/cuda_functions/nms_3D/src/cuda
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_50
cd ../../
singularity exec --nv -B /scratch:/scratch $singularity_path python3 build.py

cd ~/mrcnn3d_mdt/cuda_functions/roi_align_2D/roi_align/src/cuda 
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_50
cd ../../
singularity exec --nv -B /scratch:/scratch $singularity_path python3 build.py


cd ~/mrcnn3d_mdt/cuda_functions/roi_align_3D/roi_align/src/cuda 
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_50
cd ../../
singularity exec --nv -B /scratch:/scratch $singularity_path python3 build.py