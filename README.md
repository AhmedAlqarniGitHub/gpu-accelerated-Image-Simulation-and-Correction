# gpu-accelerated-Image-Simulation-and-Correction
Course project

# Running OpenACC Code
# to compile with explicit OpenACC Data Management, run the following code:
!nvc -fast -ta=tesla -Minfo=accel -o OpenCC OpenACC.c && ./OpenCC
# to compile with OpenACC Managed Memory, run the following code:
!nvc -fast -ta=tesla:managed -Minfo=accel -o OpenCC OpenACC.c && ./OpenCC
# to profile the code, run the following code:
!nsys profile -t OpenCC --stats=true --force-overwrite true -o OpenCC ./OpenCC

# Simulate protanopia

Flow of simulate protanopia:

1- the following arguments should be available before processing:

    a- image file name

    b- header (first 54 bytes of the image file) (getc)

    c- image size

    d- buffer ( has the image stored in it as bytes “getc” 2d array cell for each color in each pixel )

    e- bitDepth ( element at index 28 in the header)

    f- colorTable (when bitdepth ≤8, the color table will be the next 1024 bytes after header

2- in the function (simulate_cvd_protanopia):

    a- write the header in the new file as is.

    b- create new 2d array with size*3 (cells for RGB)

    c- convert each pixel “rgb” from rgb to lms (using rgb2lms) by passing the addresses

    d- send lms values to simulate_protanopia function

    e- convert new lms values back to rgb, then store them in the proper index in the output buffer or 2d array.

    f- print the values using single thread to the output file byte by byte.

    g- close the file.