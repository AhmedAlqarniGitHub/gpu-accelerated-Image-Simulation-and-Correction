#!/bin/bash

# GCC 
if command -v gcc &> /dev/null
then
    gcc -fopenmp image_processer.c -o gcc_image_processer_release -lm -DRELEASE
    gcc -fopenmp image_processer.c -o gcc_image_processer -lm
fi

# ICX
if command -v icx &> /dev/null
then
    icx -qopenmp image_processer.c -o icx_image_processer_release -DRELEASE 
    icx -qopenmp image_processer.c -o icx_image_processer
fi
