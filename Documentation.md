# gpu-accelerated-Image-Simulation-and-Correction
Course project

## Running Serial Code
To compile the Serial code, run the following command:
!nvc -Minfo=accel -o serial serial.c


To profile the Serial code, run the following commands:
%timeit !./serial
!nsys profile --stats=true --force-overwrite true -o serial ./serial



## Running OpenACC Code - Explicit Data Management
To compile with explicit OpenACC Data Management, run the following command:
!nvc -fast -ta=tesla -Minfo=accel -o OpenACC-data-management OpenACC-data-management.c && ./OpenACC-data-management


To profile the code, run the following commands:
%timeit !./OpenACC-data-management
!nsys profile -t openacc --stats=true --force-overwrite true -o OpenACC-data-management ./OpenACC-data-management

To profile the code, run the following commands:
%timeit !./OpenACC-managed-memory
!nsys profile -t openacc --stats=true --force-overwrite true -o OpenACC-managed-memory ./OpenACC-managed-memory


## Running OpenACC Code - Managed Memory
To compile with OpenACC Managed Memory, run the following command:
!nvc -fast -ta=tesla:managed -Minfo=accel -o OpenACC-managed-memory OpenACC-managed-memory.c && ./OpenACC-managed-memory


## Running CUDA C Code
To compile CUDA C, run the following command (in cuda_c folder):
!nvcc -arch=sm_70 -o cuda-c cuda.cu


To profile the code, run the following commands (in cuda_c folder):
%timeit !./cuda-c
!nsys profile --stats=true --force-overwrite=true -o cuda-c-report ./cuda-c


## Running CUDA PYTHON Code
To compile CUDA PYTHON, run the following command (in cuda_python folder):
!python ./cuda_python.py


To measure the code timing, run the following commands (in cuda_python folder):
%timeit !python ./cuda_python.py



## Simulate Protanopia
Flow of simulate protanopia:

1. The following arguments should be available before processing:
   - Image file name
   - Header (first 54 bytes of the image file) (getc)
   - Image size
   - Buffer (has the image stored in it as bytes “getc” 2d array cell for each color in each pixel)
   - BitDepth (element at index 28 in the header)
   - ColorTable (when bitdepth ≤8, the color table will be the next 1024 bytes after header)

2. In the function `simulate_cvd_protanopia`:
   - Write the header in the new file as is.
   - Create new 2d array with size*3 (cells for RGB).
   - Convert each pixel “rgb” from rgb to lms (using rgb2lms) by passing the addresses.
   - Send lms values to simulate_protanopia function.
   - Convert new lms values back to rgb, then store them in the proper index in the output buffer or 2d array.
   - Print the values using single thread to the output file byte by byte.
   - Close the file.
