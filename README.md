#Introduction
This project is developed as part of the KFUPM COE 506 Course and focuses on leveraging GPU acceleration for image processing tasks. It includes implementations for simulating and correcting color vision deficiencies (CVD) such as protanopia, deuteranopia, and tritanopia.

#Components
The repository consists of several key components:

CUDA in C: Core GPU-accelerated algorithms implemented in C using CUDA. View CUDA C Code

CUDA with Python: A Python wrapper for the CUDA C code, providing a more accessible interface. View CUDA Python Code

OpenACC: Alternative GPU acceleration using OpenACC, demonstrating different approaches to parallel computing. View OpenACC Code

Image Processing Utilities: A collection of utility functions for basic image processing tasks like converting to grayscale, adjusting brightness, etc. View Image Processing Utilities

#Usage
The project includes scripts and instructions for running and profiling the code in different modes (Serial, CUDA, OpenACC). Detailed instructions are provided in the README.

#Sample Application Flow
A typical flow for simulating protanopia involves processing an image file through various stages, including RGB to LMS conversion, simulating CVD, and converting back to RGB. The process is detailed in the README.

#License
This project is licensed under the MIT License - see the LICENSE file for details.

