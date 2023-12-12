#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

__global__ void simulate_protanopia_kernel(const float *l, const float *m, const float *s, float *rr, float *gg, float *bb, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // Simulate protanopia
        float ll = (0.0 * l[idx]) + (2.02344 * m[idx]) + (-2.52581 * s[idx]);
        float mm = (0.0 * l[idx]) + (1.0 * m[idx]) + (0.0 * s[idx]);
        float ss = (0.0 * l[idx]) + (0.0 * m[idx]) + (1.0 * s[idx]);

        // convert to srgb
        rr[idx] = (ll * 5.47221206) + (mm * -1.1252419) + (ss * 0.02980165);
        gg[idx] = (ll * -4.6419601) + (mm * 2.29317094) + (ss * -0.19318073);
        bb[idx] = (ll * 0.16963708) + (mm * -0.1678952) + (ss * 1.16364789);

        // convert to rgb
        rr[idx] = rr[idx] <= 0.0031308 ? rr[idx] * 12.92 : (pow(1.055 * rr[idx], 0.41666) - 0.055);
        gg[idx] = gg[idx] <= 0.0031308 ? gg[idx] * 12.92 : (pow(1.055 * gg[idx], 0.41666) - 0.055);
        bb[idx] = bb[idx] <= 0.0031308 ? bb[idx] * 12.92 : (pow(1.055 * bb[idx], 0.41666) - 0.055);

        rr[idx] *= 255;
        gg[idx] *= 255;
        bb[idx] *= 255;

        // clamp value
        rr[idx] = (rr[idx] < 0) ? 0 : (rr[idx] > 255) ? 255: rr[idx];
        gg[idx] = (gg[idx] < 0) ? 0 : (gg[idx] > 255) ? 255: gg[idx];
        bb[idx] = (bb[idx] < 0) ? 0 : (bb[idx] > 255) ? 255: bb[idx];
    }
}


__global__ void simulate_deuteranopia_kernel(const float *l, const float *m, const float *s, float *rr, float *gg, float *bb, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float ll = (1.0 * l[idx]) + (0.0 * m[idx]) + (0.0 * s[idx]);
        float mm = (0.494207 * l[idx]) + (0.0 * m[idx]) + (1.24827 * s[idx]);
        float ss = (0.0 * l[idx]) + (0.0 * m[idx]) + (1.0 * s[idx]);

        // convert to srgb
        rr[idx] = (ll * 5.47221206) + (mm * -1.1252419) + (ss * 0.02980165);
        gg[idx] = (ll * -4.6419601) + (mm * 2.29317094) + (ss * -0.19318073);
        bb[idx] = (ll * 0.16963708) + (mm * -0.1678952) + (ss * 1.16364789);

        // convert to rgb
        rr[idx] = rr[idx] <= 0.0031308 ? rr[idx] * 12.92 : (pow(1.055 * rr[idx], 0.41666) - 0.055);
        gg[idx] = gg[idx] <= 0.0031308 ? gg[idx] * 12.92 : (pow(1.055 * gg[idx], 0.41666) - 0.055);
        bb[idx] = bb[idx] <= 0.0031308 ? bb[idx] * 12.92 : (pow(1.055 * bb[idx], 0.41666) - 0.055);

        rr[idx] *= 255;
        gg[idx] *= 255;
        bb[idx] *= 255;

        // clamp value
        rr[idx] = (rr[idx] < 0) ? 0 : (rr[idx] > 255) ? 255: rr[idx];
        gg[idx] = (gg[idx] < 0) ? 0 : (gg[idx] > 255) ? 255: gg[idx];
        bb[idx] = (bb[idx] < 0) ? 0 : (bb[idx] > 255) ? 255: bb[idx];
    }

}

__global__ void simulate_tritanopia_kernel(const float *l, const float *m, const float *s, float *rr, float *gg, float *bb, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float ll = (1.0 * l[idx]) + (0.0 * m[idx]) + (0.0 * s[idx]);
        float mm = (0.0 * l[idx]) + (1.0 * m[idx]) + (0.0 * s[idx]);
        float ss = (-0.395913 * l[idx]) + (0.801109 * m[idx]) + (0.0 * s[idx]);

        // convert to srgb
        rr[idx] = (ll * 5.47221206) + (mm * -1.1252419) + (ss * 0.02980165);
        gg[idx] = (ll * -4.6419601) + (mm * 2.29317094) + (ss * -0.19318073);
        bb[idx] = (ll * 0.16963708) + (mm * -0.1678952) + (ss * 1.16364789);

        // convert to rgb
        rr[idx] = rr[idx] <= 0.0031308 ? rr[idx] * 12.92 : (pow(1.055 * rr[idx], 0.41666) - 0.055);
        gg[idx] = gg[idx] <= 0.0031308 ? gg[idx] * 12.92 : (pow(1.055 * gg[idx], 0.41666) - 0.055);
        bb[idx] = bb[idx] <= 0.0031308 ? bb[idx] * 12.92 : (pow(1.055 * bb[idx], 0.41666) - 0.055);

        rr[idx] *= 255;
        gg[idx] *= 255;
        bb[idx] *= 255;

        // clamp value
        rr[idx] = (rr[idx] < 0) ? 0 : (rr[idx] > 255) ? 255: rr[idx];
        gg[idx] = (gg[idx] < 0) ? 0 : (gg[idx] > 255) ? 255: gg[idx];
        bb[idx] = (bb[idx] < 0) ? 0 : (bb[idx] > 255) ? 255: bb[idx];
    }
}

__global__ void correct_protanopia_kernel(const float *l, const float *m, const float *s, float *rr, float *gg, float *bb, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Apply protanopia correction
        float ll = (0.0 * l[idx]) + (2.02344 * m[idx]) + (-2.52581 * s[idx]);
        float mm = (0.0 * l[idx]) + (1.0 * m[idx]) + (0.0 * s[idx]);
        float ss = (0.0 * l[idx]) + (0.0 * m[idx]) + (1.0 * s[idx]);

        // Convert corrected LMS back to linear RGB
        float linearR = (ll * 5.47221206) + (mm * -4.6419601) + (ss * 0.16963708);
        float linearG = (ll * -1.1252419) + (mm * 2.29317094) + (ss * -0.1678952);
        float linearB = (ll * 0.02980165) + (mm * -0.19318073) + (ss * 1.16364789);

        // Convert to sRGB
        rr[idx] = linearR <= 0.0031308 ? linearR * 12.92 : (pow((1.055 * linearR), (1.0 / 2.4)) - 0.055);
        gg[idx] = linearG <= 0.0031308 ? linearG * 12.92 : (pow((1.055 * linearG), (1.0 / 2.4)) - 0.055);
        bb[idx] = linearB <= 0.0031308 ? linearB * 12.92 : (pow((1.055 * linearB), (1.0 / 2.4)) - 0.055);

        // Scale to 255 and clamp
        rr[idx] = fminf(fmaxf(rr[idx] * 255.0, 0.0), 255.0);
        gg[idx] = fminf(fmaxf(gg[idx] * 255.0, 0.0), 255.0);
        bb[idx] = fminf(fmaxf(bb[idx] * 255.0, 0.0), 255.0);
    }
}



__global__ void correct_deuteranopia_kernel(const float *l, const float *m, const float *s, float *rr, float *gg, float *bb, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Apply deuteranopia correction
        float ll = (1.0 * l[idx]) + (0.0 * m[idx]) + (0.0 * s[idx]);
        float mm = (0.494207 * l[idx]) + (0.0 * m[idx]) + (1.24827 * s[idx]);
        float ss = (0.0 * l[idx]) + (0.0 * m[idx]) + (1.0 * s[idx]);

        // Convert corrected LMS back to linear RGB
        float linearR = (ll * 5.47221206) + (mm * -4.6419601) + (ss * 0.16963708);
        float linearG = (ll * -1.1252419) + (mm * 2.29317094) + (ss * -0.1678952);
        float linearB = (ll * 0.02980165) + (mm * -0.19318073) + (ss * 1.16364789);

        // Convert to sRGB
        rr[idx] = linearR <= 0.0031308 ? linearR * 12.92 : (pow(1.055 * linearR, 1/2.4) - 0.055);
        gg[idx] = linearG <= 0.0031308 ? linearG * 12.92 : (pow(1.055 * linearG, 1/2.4) - 0.055);
        bb[idx] = linearB <= 0.0031308 ? linearB * 12.92 : (pow(1.055 * linearB, 1/2.4) - 0.055);

        // Scale to 255 and clamp
        rr[idx] = fminf(fmaxf(rr[idx] * 255.0, 0.0), 255.0);
        gg[idx] = fminf(fmaxf(gg[idx] * 255.0, 0.0), 255.0);
        bb[idx] = fminf(fmaxf(bb[idx] * 255.0, 0.0), 255.0);
    }
}



__global__ void correct_tritanopia_kernel(const float *l, const float *m, const float *s, float *rr, float *gg, float *bb, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Apply tritanopia correction
        float ll = (1.0 * l[idx]) + (0.0 * m[idx]) + (0.0 * s[idx]);
        float mm = (0.0 * l[idx]) + (1.0 * m[idx]) + (0.0 * s[idx]);
        float ss = (-0.395913 * l[idx]) + (0.801109 * m[idx]) + (0.0 * s[idx]);

        // Convert corrected LMS back to linear RGB
        float linearR = (ll * 5.47221206) + (mm * -4.6419601) + (ss * 0.16963708);
        float linearG = (ll * -1.1252419) + (mm * 2.29317094) + (ss * -0.1678952);
        float linearB = (ll * 0.02980165) + (mm * -0.19318073) + (ss * 1.16364789);

        // Convert to sRGB
        rr[idx] = linearR <= 0.0031308 ? linearR * 12.92 : (pow(1.055 * linearR, 1/2.4) - 0.055);
        gg[idx] = linearG <= 0.0031308 ? linearG * 12.92 : (pow(1.055 * linearG, 1/2.4) - 0.055);
        bb[idx] = linearB <= 0.0031308 ? linearB * 12.92 : (pow(1.055 * linearB, 1/2.4) - 0.055);

        // Scale to 255 and clamp
        rr[idx] = fminf(fmaxf(rr[idx] * 255.0, 0.0), 255.0);
        gg[idx] = fminf(fmaxf(gg[idx] * 255.0, 0.0), 255.0);
        bb[idx] = fminf(fmaxf(bb[idx] * 255.0, 0.0), 255.0);
    }
}

void processImageWithCUDA(const char *imageFileName, const char *outputFileName)
{
    char ImageFilePath[150];
    sprintf(ImageFilePath, "images/%s.bmp", imageFileName);

    printf("******** This code is executing the colored image processing applications ***** \n");
    printf(" ==  %s \n", ImageFilePath);
    FILE *fIn = fopen(ImageFilePath, "r"); // Input File name

    unsigned char header[54];
    unsigned char colorTable[1024];
    int i, j;

    if (fIn == NULL) // check if the input file has not been opened succesfully.
    {
        printf("File does not exist.\n");
    }

    for (i = 0; i < 54; i++) // read the 54 byte header from fIn
    {
        header[i] = getc(fIn);
    }

    int height = *(int *)&header[18];
    int width = *(int *)&header[22];
    int bitDepth = *(int *)&header[28];
    int size = width * height;

    if (bitDepth <= 8) // if ColorTable present, extract it.
    {
        fread(colorTable, sizeof(unsigned char), 1024, fIn);
    }

    unsigned char *buffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char)); // Allocate memory for color components (RGB)
    fread(buffer, sizeof(unsigned char), size * 3, fIn);                               // Read the image data
    fclose(fIn);

    // Allocate host memory for LMS components
    float *l = (float *)malloc(size * sizeof(float));
    float *m = (float *)malloc(size * sizeof(float));
    float *s = (float *)malloc(size * sizeof(float));

    // Allocate host memory for LMS components
    float *r_protanopia = (float *)malloc(size * sizeof(float));
    float *g_protanopia = (float *)malloc(size * sizeof(float));
    float *b_protanopia = (float *)malloc(size * sizeof(float));

    // Allocate host memory for LMS components
    float *r_deuteranopia = (float *)malloc(size * sizeof(float));
    float *g_deuteranopia = (float *)malloc(size * sizeof(float));
    float *b_deuteranopia = (float *)malloc(size * sizeof(float));

    // Allocate host memory for LMS components
    float *r_tritanopia = (float *)malloc(size * sizeof(float));
    float *g_tritanopia = (float *)malloc(size * sizeof(float));
    float *b_tritanopia = (float *)malloc(size * sizeof(float));

    float *r_corrected_protanopia = (float *)malloc(size * sizeof(float));
    float *g_corrected_protanopia = (float *)malloc(size * sizeof(float));
    float *b_corrected_protanopia = (float *)malloc(size * sizeof(float));

    float *r_corrected_deuteranopia = (float *)malloc(size * sizeof(float));
    float *g_corrected_deuteranopia = (float *)malloc(size * sizeof(float));
    float *b_corrected_deuteranopia = (float *)malloc(size * sizeof(float));

    float *r_corrected_tritanopia = (float *)malloc(size * sizeof(float));
    float *g_corrected_tritanopia = (float *)malloc(size * sizeof(float));
    float *b_corrected_tritanopia = (float *)malloc(size * sizeof(float));

    // Convert RGB to LMS (Host)
    for (int i = 0; i < size; i++)
    {
        // Assuming the buffer is in RGB format
        unsigned char r = buffer[i * 3];
        unsigned char g = buffer[i * 3 + 1];
        unsigned char b = buffer[i * 3 + 2];

        // Convert to LMS - insert the rgb2lms conversion logic here
        float rr = (r / 255.0);
        float gg = (g / 255.0);
        float bb = (b / 255.0);

        // convert to srgb
        rr = rr <= 0.04045 ? (rr) / 12.92 : pow((rr + 0.055) / 1.055, 2.4);
        gg = gg <= 0.04045 ? (gg) / 12.92 : pow((gg + 0.055) / 1.055, 2.4);
        bb = bb <= 0.04045 ? (bb) / 12.92 : pow((bb + 0.055) / 1.055, 2.4);

        // convert to lms
        l[i]  = (rr * 0.31399022) + (gg * 0.15537241) + (bb * 0.01775239);
        m[i]  = (rr * 0.63951294) + (gg * 0.75789446) + (bb * 0.10944209);
        s[i]  = (rr * 0.04649755) + (gg * 0.08670142) + (bb * 0.87256922);
    }
    
    // Create CUDA streams
    cudaStream_t streamProtanopia, streamDeuteranopia, streamTritanopia;
    cudaStream_t streamCorrectedProtanopia, streamCorrectedDeuteranopia, streamCorrectedTritanopia;

    cudaStreamCreate(&streamProtanopia);
    cudaStreamCreate(&streamDeuteranopia);
    cudaStreamCreate(&streamTritanopia);
    cudaStreamCreate(&streamCorrectedProtanopia);
    cudaStreamCreate(&streamCorrectedDeuteranopia);
    cudaStreamCreate(&streamCorrectedTritanopia);

    
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory
    float *dev_l, *dev_m, *dev_s;
    float *dev_rr_protanopia, *dev_gg_protanopia, *dev_bb_protanopia;
    float *dev_rr_deuteranopia, *dev_gg_deuteranopia, *dev_bb_deuteranopia;
    float *dev_rr_tritanopia, *dev_gg_tritanopia, *dev_bb_tritanopia;

    // Allocate device memory for corrected LMS values
    float *dev_rr_corrected_protanopia, *dev_gg_corrected_protanopia, *dev_bb_corrected_protanopia;
    float *dev_rr_corrected_deuteranopia, *dev_gg_corrected_deuteranopia, *dev_bb_corrected_deuteranopia;
    float *dev_rr_corrected_tritanopia, *dev_gg_corrected_tritanopia, *dev_bb_corrected_tritanopia;
    

    cudaMalloc((void **)&dev_l, size * sizeof(float));
    cudaMalloc((void **)&dev_m, size * sizeof(float));
    cudaMalloc((void **)&dev_s, size * sizeof(float));

    cudaMalloc((void **)&dev_rr_protanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_gg_protanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_bb_protanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_rr_corrected_protanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_gg_corrected_protanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_bb_corrected_protanopia, size * sizeof(float));

    cudaMalloc((void **)&dev_rr_deuteranopia, size * sizeof(float));
    cudaMalloc((void **)&dev_gg_deuteranopia, size * sizeof(float));
    cudaMalloc((void **)&dev_bb_deuteranopia, size * sizeof(float));
    cudaMalloc((void **)&dev_rr_corrected_deuteranopia, size * sizeof(float));
    cudaMalloc((void **)&dev_gg_corrected_deuteranopia, size * sizeof(float));
    cudaMalloc((void **)&dev_bb_corrected_deuteranopia, size * sizeof(float));
    
    cudaMalloc((void **)&dev_rr_tritanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_gg_tritanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_bb_tritanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_rr_corrected_tritanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_gg_corrected_tritanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_bb_corrected_tritanopia, size * sizeof(float));

    // Copy data from host to device in both streams
    cudaMemcpyAsync(dev_l, l, size * sizeof(float), cudaMemcpyHostToDevice, streamProtanopia);
    cudaMemcpyAsync(dev_m, m, size * sizeof(float), cudaMemcpyHostToDevice, streamProtanopia);
    cudaMemcpyAsync(dev_s, s, size * sizeof(float), cudaMemcpyHostToDevice, streamProtanopia);

    cudaMemcpyAsync(dev_l, l, size * sizeof(float), cudaMemcpyHostToDevice, streamDeuteranopia);
    cudaMemcpyAsync(dev_m, m, size * sizeof(float), cudaMemcpyHostToDevice, streamDeuteranopia);
    cudaMemcpyAsync(dev_s, s, size * sizeof(float), cudaMemcpyHostToDevice, streamDeuteranopia);

    cudaMemcpyAsync(dev_l, l, size * sizeof(float), cudaMemcpyHostToDevice, streamTritanopia);
    cudaMemcpyAsync(dev_m, m, size * sizeof(float), cudaMemcpyHostToDevice, streamTritanopia);
    cudaMemcpyAsync(dev_s, s, size * sizeof(float), cudaMemcpyHostToDevice, streamTritanopia);

    cudaMemcpyAsync(dev_l, l, size * sizeof(float), cudaMemcpyHostToDevice, streamCorrectedProtanopia);
    cudaMemcpyAsync(dev_m, m, size * sizeof(float), cudaMemcpyHostToDevice, streamCorrectedProtanopia);
    cudaMemcpyAsync(dev_s, s, size * sizeof(float), cudaMemcpyHostToDevice, streamCorrectedProtanopia);
    
    cudaMemcpyAsync(dev_l, l, size * sizeof(float), cudaMemcpyHostToDevice, streamCorrectedDeuteranopia);
    cudaMemcpyAsync(dev_m, m, size * sizeof(float), cudaMemcpyHostToDevice, streamCorrectedDeuteranopia);
    cudaMemcpyAsync(dev_s, s, size * sizeof(float), cudaMemcpyHostToDevice, streamCorrectedDeuteranopia);

    cudaMemcpyAsync(dev_l, l, size * sizeof(float), cudaMemcpyHostToDevice, streamCorrectedTritanopia);
    cudaMemcpyAsync(dev_m, m, size * sizeof(float), cudaMemcpyHostToDevice, streamCorrectedTritanopia);
    cudaMemcpyAsync(dev_s, s, size * sizeof(float), cudaMemcpyHostToDevice, streamCorrectedTritanopia);


    // Process 1: simulate_protanopia_kernel in streamProtanopia, streamCorrectedProtanopia
    simulate_protanopia_kernel<<<blocksPerGrid, threadsPerBlock, 0, streamProtanopia>>>(dev_l, dev_m, dev_s, dev_rr_protanopia, dev_gg_protanopia, dev_bb_protanopia, size);
    // Check for errors immediately after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Failed to launch simulate_protanopia_kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    correct_protanopia_kernel<<<blocksPerGrid, threadsPerBlock, 0, streamCorrectedProtanopia>>>(dev_l, dev_m, dev_s, dev_rr_corrected_protanopia, dev_gg_corrected_protanopia, dev_bb_corrected_protanopia, size);
    // Check for errors immediately after kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Failed to launch correct_protanopia_kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Process 2: simulate_deuteranopia_kernel in streamDeuteranopia, streamCorrectedDeuteranopia
    simulate_deuteranopia_kernel<<<blocksPerGrid, threadsPerBlock, 0, streamDeuteranopia>>>(dev_l, dev_m, dev_s, dev_rr_deuteranopia, dev_gg_deuteranopia, dev_bb_deuteranopia, size);
    // Check for errors immediately after kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Failed to launch simulate_deuteranopia_kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    correct_deuteranopia_kernel<<<blocksPerGrid, threadsPerBlock, 0, streamCorrectedDeuteranopia>>>(dev_l, dev_m, dev_s, dev_rr_corrected_deuteranopia, dev_gg_corrected_deuteranopia, dev_bb_corrected_deuteranopia, size);
    // Check for errors immediately after kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Failed to launch correct_deuteranopia_kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Process 3: simulate_tritanopia_kernel in streamTritanopia, streamCorrectedTritanopia
    simulate_tritanopia_kernel<<<blocksPerGrid, threadsPerBlock, 0, streamTritanopia>>>(dev_l, dev_m, dev_s, dev_rr_tritanopia, dev_gg_tritanopia, dev_bb_tritanopia, size);
    // Check for errors immediately after kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Failed to launch simulate_tritanopia_kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    correct_tritanopia_kernel<<<blocksPerGrid, threadsPerBlock, 0, streamCorrectedTritanopia>>>(dev_l, dev_m, dev_s, dev_rr_corrected_tritanopia, dev_gg_corrected_tritanopia, dev_bb_corrected_tritanopia, size);
    // Check for errors immediately after kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Failed to launch correct_tritanopia_kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Allocate host buffers for the output
    unsigned char *protanopiaBuffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char));
    unsigned char *deuteranopiaBuffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char));
    unsigned char *tritanopiaBuffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char));
    // Convert corrected LMS back to RGB and write to file
    unsigned char *corrected_protanopiaBuffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char));
    unsigned char *corrected_deuteranopiaBuffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char));
    unsigned char *corrected_tritanopiaBuffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char));


    // Wait for streamProtanopia to finish
    cudaStreamSynchronize(streamProtanopia);
    // Wait for streamProtanopia to finish
    cudaStreamSynchronize(streamCorrectedProtanopia);

    // Copy results back to host in streamProtanopia
    cudaMemcpyAsync(r_protanopia, dev_rr_protanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamProtanopia);
    cudaMemcpyAsync(g_protanopia, dev_gg_protanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamProtanopia);
    cudaMemcpyAsync(b_protanopia, dev_bb_protanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamProtanopia);
    cudaMemcpyAsync(r_corrected_protanopia, dev_rr_corrected_protanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamCorrectedProtanopia);
    cudaMemcpyAsync(g_corrected_protanopia, dev_gg_corrected_protanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamCorrectedProtanopia);
    cudaMemcpyAsync(b_corrected_protanopia, dev_bb_corrected_protanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamCorrectedProtanopia);

    // Wait for streamDeuteranopia to finish
    cudaStreamSynchronize(streamDeuteranopia);
    // Wait for streamDeuteranopia to finish
    cudaStreamSynchronize(streamCorrectedDeuteranopia);

    // Copy results back to host in streamDeuteranopia
    cudaMemcpyAsync(r_deuteranopia, dev_rr_deuteranopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamDeuteranopia);
    cudaMemcpyAsync(g_deuteranopia, dev_gg_deuteranopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamDeuteranopia);
    cudaMemcpyAsync(b_deuteranopia, dev_bb_deuteranopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamDeuteranopia);
    cudaMemcpyAsync(r_corrected_deuteranopia, dev_rr_corrected_deuteranopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamCorrectedDeuteranopia);
    cudaMemcpyAsync(g_corrected_deuteranopia, dev_gg_corrected_deuteranopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamCorrectedDeuteranopia);
    cudaMemcpyAsync(b_corrected_deuteranopia, dev_bb_corrected_deuteranopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamCorrectedDeuteranopia);


    // Wait for streamTritanopia to finish
    cudaStreamSynchronize(streamTritanopia);
    // Wait for streamTritanopia to finish
    cudaStreamSynchronize(streamCorrectedTritanopia);


    // Copy results back to host in streamTritanopia
    cudaMemcpyAsync(r_tritanopia, dev_rr_tritanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamTritanopia);
    cudaMemcpyAsync(g_tritanopia, dev_gg_tritanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamTritanopia);
    cudaMemcpyAsync(b_tritanopia, dev_bb_tritanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamTritanopia);
    cudaMemcpyAsync(r_corrected_tritanopia, dev_rr_corrected_tritanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamCorrectedTritanopia);
    cudaMemcpyAsync(g_corrected_tritanopia, dev_gg_corrected_tritanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamCorrectedTritanopia);
    cudaMemcpyAsync(b_corrected_tritanopia, dev_bb_corrected_tritanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamCorrectedTritanopia);


    // Convert LMS back to RGB for protanopia and write to file
    for (int i = 0; i < size; i++)
    {
        protanopiaBuffer[i * 3] = static_cast<unsigned char>(r_protanopia[i]);
        protanopiaBuffer[i * 3 + 1] = static_cast<unsigned char>(g_protanopia[i]);
        protanopiaBuffer[i * 3 + 2] = static_cast<unsigned char>(b_protanopia[i]);

        deuteranopiaBuffer[i * 3] = static_cast<unsigned char>(r_deuteranopia[i]);
        deuteranopiaBuffer[i * 3 + 1] = static_cast<unsigned char>(g_deuteranopia[i]);
        deuteranopiaBuffer[i * 3 + 2] = static_cast<unsigned char>(b_deuteranopia[i]);

        tritanopiaBuffer[i * 3] = static_cast<unsigned char>(r_tritanopia[i]);
        tritanopiaBuffer[i * 3 + 1] = static_cast<unsigned char>(g_tritanopia[i]);
        tritanopiaBuffer[i * 3 + 2] = static_cast<unsigned char>(b_tritanopia[i]);

        corrected_protanopiaBuffer[i * 3]     = static_cast<unsigned char>(r_corrected_protanopia[i]);
        corrected_protanopiaBuffer[i * 3 + 1] = static_cast<unsigned char>(g_corrected_protanopia[i]);
        corrected_protanopiaBuffer[i * 3 + 2] = static_cast<unsigned char>(b_corrected_protanopia[i]);

        corrected_deuteranopiaBuffer[i * 3] = static_cast<unsigned char>(r_corrected_deuteranopia[i]);
        corrected_deuteranopiaBuffer[i * 3 + 1] = static_cast<unsigned char>(g_corrected_deuteranopia[i]);
        corrected_deuteranopiaBuffer[i * 3 + 2] = static_cast<unsigned char>(b_corrected_deuteranopia[i]);

        corrected_tritanopiaBuffer[i * 3] = static_cast<unsigned char>(r_corrected_tritanopia[i]);
        corrected_tritanopiaBuffer[i * 3 + 1] = static_cast<unsigned char>(g_corrected_tritanopia[i]);
        corrected_tritanopiaBuffer[i * 3 + 2] = static_cast<unsigned char>(b_corrected_tritanopia[i]);
    }

    char protanopiaFileName[150];
    sprintf(protanopiaFileName, "%s_protanopia.bmp", outputFileName);
    FILE *fOut = fopen(protanopiaFileName, "wb");
    fwrite(header, sizeof(unsigned char), 54, fOut);
    fwrite(protanopiaBuffer, sizeof(unsigned char), size * 3, fOut);
    fclose(fOut);

    char deuteranopiaFileName[150];
    sprintf(deuteranopiaFileName, "%s_deuteranopia.bmp", outputFileName);
    FILE *fOut2 = fopen(deuteranopiaFileName, "wb");
    fwrite(header, sizeof(unsigned char), 54, fOut2);
    fwrite(deuteranopiaBuffer, sizeof(unsigned char), size * 3, fOut2);
    fclose(fOut2);

    char tritanopiaFileName[150];
    sprintf(tritanopiaFileName, "%s_tritanopia.bmp", outputFileName);
    FILE *fOut3 = fopen(tritanopiaFileName, "wb");
    fwrite(header, sizeof(unsigned char), 54, fOut3);
    fwrite(tritanopiaBuffer, sizeof(unsigned char), size * 3, fOut3);
    fclose(fOut3);

    char corrected_protanopiaFileName[150];
    sprintf(corrected_protanopiaFileName, "%s_corrected_protanopia.bmp", outputFileName);
    FILE *fOutCorrectedProtanopia = fopen(corrected_protanopiaFileName, "wb");
    fwrite(header, sizeof(unsigned char), 54, fOutCorrectedProtanopia);
    fwrite(corrected_protanopiaBuffer, sizeof(unsigned char), size * 3, fOutCorrectedProtanopia);
    fclose(fOutCorrectedProtanopia);

    char corrected_deuteranopiaFileName[150];
    sprintf(corrected_deuteranopiaFileName, "%s_corrected_deuteranopia.bmp", outputFileName);
    FILE *fOutCorrectedDeuteranopia = fopen(corrected_deuteranopiaFileName, "wb");
    fwrite(header, sizeof(unsigned char), 54, fOutCorrectedDeuteranopia);
    fwrite(corrected_deuteranopiaBuffer, sizeof(unsigned char), size * 3, fOutCorrectedDeuteranopia);
    fclose(fOutCorrectedDeuteranopia);

    char corrected_tritanopiaFileName[150];
    sprintf(corrected_tritanopiaFileName, "%s_corrected_tritanopia.bmp", outputFileName);
    FILE *fOutCorrectedTritanopia = fopen(corrected_tritanopiaFileName, "wb");
    fwrite(header, sizeof(unsigned char), 54, fOutCorrectedTritanopia);
    fwrite(corrected_tritanopiaBuffer, sizeof(unsigned char), size * 3, fOutCorrectedTritanopia);
    fclose(fOutCorrectedTritanopia);

    // Cleanup
    cudaStreamDestroy(streamProtanopia);
    cudaStreamDestroy(streamDeuteranopia);
    cudaStreamDestroy(streamTritanopia);
    cudaStreamDestroy(streamCorrectedProtanopia);
    cudaStreamDestroy(streamCorrectedDeuteranopia);
    cudaStreamDestroy(streamCorrectedTritanopia);

    free(protanopiaBuffer);
    free(deuteranopiaBuffer);
    free(tritanopiaBuffer);

    free(corrected_protanopiaBuffer);
    free(corrected_deuteranopiaBuffer);
    free(corrected_tritanopiaBuffer);

    free(buffer);

    free(r_protanopia);
    free(g_protanopia);
    free(b_protanopia);

    free(r_deuteranopia);
    free(g_deuteranopia);
    free(b_deuteranopia);

    free(r_tritanopia);
    free(g_tritanopia);
    free(b_tritanopia);

    free(r_corrected_protanopia);
    free(g_corrected_protanopia);
    free(b_corrected_protanopia);

    free(r_corrected_deuteranopia);
    free(g_corrected_deuteranopia);
    free(b_corrected_deuteranopia);

    free(r_corrected_tritanopia);
    free(g_corrected_tritanopia);
    free(b_corrected_tritanopia);

    cudaFree(dev_l);
    cudaFree(dev_m);
    cudaFree(dev_s);
    
    cudaFree(dev_rr_protanopia);
    cudaFree(dev_gg_protanopia);
    cudaFree(dev_bb_protanopia);

    cudaFree(dev_rr_deuteranopia);
    cudaFree(dev_gg_deuteranopia);
    cudaFree(dev_bb_deuteranopia);

    cudaFree(dev_rr_tritanopia);
    cudaFree(dev_gg_tritanopia);
    cudaFree(dev_bb_tritanopia);

    cudaFree(dev_rr_corrected_protanopia);
    cudaFree(dev_gg_corrected_protanopia);
    cudaFree(dev_bb_corrected_protanopia);

    cudaFree(dev_rr_corrected_deuteranopia);
    cudaFree(dev_gg_corrected_deuteranopia);
    cudaFree(dev_bb_corrected_deuteranopia);

    cudaFree(dev_rr_corrected_tritanopia);
    cudaFree(dev_gg_corrected_tritanopia);
    cudaFree(dev_bb_corrected_tritanopia);
}

int main(int argc, char *argv[])
{
    processImageWithCUDA("lena_color", "temp");
}