#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

__global__ void rgb2lms_kernel(const unsigned char *r, const unsigned char *g, const unsigned char *b, float *l, float *m, float *s, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float rr = r[idx] / 255.0f;
        float gg = g[idx] / 255.0f;
        float bb = b[idx] / 255.0f;

        // Conversion code as in your function
        // convert to srgb
        rr = rr <= 0.04045 ? (rr) / 12.92 : pow((rr + 0.055) / 1.055, 2.4);
        gg = gg <= 0.04045 ? (gg) / 12.92 : pow((gg + 0.055) / 1.055, 2.4);
        bb = bb <= 0.04045 ? (bb) / 12.92 : pow((bb + 0.055) / 1.055, 2.4);

        l[idx] = (rr * 0.31399022) + (gg * 0.15537241) + (bb * 0.01775239);
        m[idx] = (rr * 0.63951294) + (gg * 0.75789446) + (bb * 0.10944209);
        s[idx] = (rr * 0.04649755) + (gg * 0.08670142) + (bb * 0.87256922);
    }
}

__global__ void simulate_protanopia_kernel(const float *l, const float *m, const float *s, float *ll, float *mm, float *ss, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        ll[idx] = (0.0 * l[idx]) + (2.02344 * m[idx]) + (-2.52581 * s[idx]);
        mm[idx] = (0.0 * l[idx]) + (1.0 * m[idx]) + (0.0 * s[idx]);
        ss[idx] = (0.0 * l[idx]) + (0.0 * m[idx]) + (1.0 * s[idx]);
    }
}

__global__ void simulate_deuteranopia_kernel(const float *l, const float *m, const float *s, float *ll, float *mm, float *ss, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        ll[idx] = (1.0 * l[idx]) + (0.0 * m[idx]) + (0.0 * s[idx]);
        mm[idx] = (0.494207 * l[idx]) + (0.0 * m[idx]) + (1.24827 * s[idx]);
        ss[idx] = (0.0 * l[idx]) + (0.0 * m[idx]) + (1.0 * s[idx]);
    }

}

__global__ void simulate_tritanopia_kernel(const float *l, const float *m, const float *s, float *ll, float *mm, float *ss, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        ll[idx] = (0.0 * l[idx]) + (2.02344 * m[idx]) + (-2.52581 * s[idx]);
        mm[idx] = (0.0 * l[idx]) + (1.0 * m[idx]) + (0.0 * s[idx]);
        ss[idx] = (0.0 * l[idx]) + (0.0 * m[idx]) + (1.0 * s[idx]);
    }
}

__global__ void correct_protanopia_kernel(const float *l, const float *m, const float *s, float *ll, float *mm, float *ss, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        ll[idx] = (0.0 * l[idx]) + (2.02344 * m[idx]) + (-2.52581 * s[idx]);
        mm[idx] = (0.0 * l[idx]) + (1.0 * m[idx]) + (0.0 * s[idx]);
        ss[idx] = (0.0 * l[idx]) + (0.0 * m[idx]) + (1.0 * s[idx]);
    }
}

__global__ void correct_deuteranopia_kernel(const float *l, const float *m, const float *s, float *ll, float *mm, float *ss, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        ll[idx] = (1.0 * l[idx]) + (0.0 * m[idx]) + (0.0 * s[idx]);
        mm[idx] = (0.494207 * l[idx]) + (0.0 * m[idx]) + (1.24827 * s[idx]);
        ss[idx] = (0.0 * l[idx]) + (0.0 * m[idx]) + (1.0 * s[idx]);
    }
}

__global__ void correct_tritanopia_kernel(const float *l, const float *m, const float *s, float *ll, float *mm, float *ss, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        ll[idx] = (1.0 * l[idx]) + (0.0 * m[idx]) + (0.0 * s[idx]);
        mm[idx] = (0.0 * l[idx]) + (1.0 * m[idx]) + (0.0 * s[idx]);
        ss[idx] = (-0.395913 * l[idx]) + (0.801109 * m[idx]) + (0.0 * s[idx]);
    }
}


void lms2rgb(float l, float m, float s,
             unsigned char *r, unsigned char *g, unsigned char *b)
{

    // convert to srgb
    float rr = (l * 5.47221206) + (m * -1.1252419) + (s * 0.02980165);
    float gg = (l * -4.6419601) + (m * 2.29317094) + (s * -0.19318073);
    float bb = (l * 0.16963708) + (m * -0.1678952) + (s * 1.16364789);

    // convert to rgb
    rr = rr <= 0.0031308 ? rr * 12.92 : (pow(1.055 * rr, 0.41666) - 0.055);
    gg = gg <= 0.0031308 ? gg * 12.92 : (pow(1.055 * gg, 0.41666) - 0.055);
    bb = bb <= 0.0031308 ? bb * 12.92 : (pow(1.055 * bb, 0.41666) - 0.055);

    rr *= 255;
    gg *= 255;
    bb *= 255;

    // clamp value
    *r = (rr < 0) ? 0 : (rr > 255) ? 255
                                   : rr;
    *g = (gg < 0) ? 0 : (gg > 255) ? 255
                                   : gg;
    *b = (bb < 0) ? 0 : (bb > 255) ? 255
                                   : bb;
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
    float *l_protanopia = (float *)malloc(size * sizeof(float));
    float *m_protanopia = (float *)malloc(size * sizeof(float));
    float *s_protanopia = (float *)malloc(size * sizeof(float));

    // Allocate host memory for LMS components
    float *l_deuteranopia = (float *)malloc(size * sizeof(float));
    float *m_deuteranopia = (float *)malloc(size * sizeof(float));
    float *s_deuteranopia = (float *)malloc(size * sizeof(float));

    // Allocate host memory for LMS components
    float *l_tritanopia = (float *)malloc(size * sizeof(float));
    float *m_tritanopia = (float *)malloc(size * sizeof(float));
    float *s_tritanopia = (float *)malloc(size * sizeof(float));

    float *l_corrected_protanopia = (float *)malloc(size * sizeof(float));
    float *m_corrected_protanopia = (float *)malloc(size * sizeof(float));
    float *s_corrected_protanopia = (float *)malloc(size * sizeof(float));

    float *l_corrected_deuteranopia = (float *)malloc(size * sizeof(float));
    float *m_corrected_deuteranopia = (float *)malloc(size * sizeof(float));
    float *s_corrected_deuteranopia = (float *)malloc(size * sizeof(float));

    float *l_corrected_tritanopia = (float *)malloc(size * sizeof(float));
    float *m_corrected_tritanopia = (float *)malloc(size * sizeof(float));
    float *s_corrected_tritanopia = (float *)malloc(size * sizeof(float));

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
    cudaStreamCreate(&streamProtanopia);
    cudaStreamCreate(&streamDeuteranopia);
    cudaStreamCreate(&streamTritanopia);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory
    float *dev_l, *dev_m, *dev_s;
    float *dev_ll_protanopia, *dev_mm_protanopia, *dev_ss_protanopia;
    float *dev_ll_deuteranopia, *dev_mm_deuteranopia, *dev_ss_deuteranopia;
    float *dev_ll_tritanopia, *dev_mm_tritanopia, *dev_ss_tritanopia;

    // Allocate device memory for corrected LMS values
    float *dev_ll_corrected_protanopia, *dev_mm_corrected_protanopia, *dev_ss_corrected_protanopia;
    float *dev_ll_corrected_deuteranopia, *dev_mm_corrected_deuteranopia, *dev_ss_corrected_deuteranopia;
    float *dev_ll_corrected_tritanopia, *dev_mm_corrected_tritanopia, *dev_ss_corrected_tritanopia;

    cudaMalloc((void **)&dev_l, size * sizeof(float));
    cudaMalloc((void **)&dev_m, size * sizeof(float));
    cudaMalloc((void **)&dev_s, size * sizeof(float));

    cudaMalloc((void **)&dev_ll_protanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_mm_protanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_ss_protanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_ll_corrected_protanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_mm_corrected_protanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_ss_corrected_protanopia, size * sizeof(float));

    cudaMalloc((void **)&dev_ll_deuteranopia, size * sizeof(float));
    cudaMalloc((void **)&dev_mm_deuteranopia, size * sizeof(float));
    cudaMalloc((void **)&dev_ss_deuteranopia, size * sizeof(float));
    cudaMalloc((void **)&dev_ll_corrected_deuteranopia, size * sizeof(float));
    cudaMalloc((void **)&dev_mm_corrected_deuteranopia, size * sizeof(float));
    cudaMalloc((void **)&dev_ss_corrected_deuteranopia, size * sizeof(float));
    
    cudaMalloc((void **)&dev_ll_tritanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_mm_tritanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_ss_tritanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_ll_corrected_tritanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_mm_corrected_tritanopia, size * sizeof(float));
    cudaMalloc((void **)&dev_ss_corrected_tritanopia, size * sizeof(float));

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

    // Process 1: simulate_protanopia_kernel in streamProtanopia
    simulate_protanopia_kernel<<<blocksPerGrid, threadsPerBlock, 0, streamProtanopia>>>(dev_l, dev_m, dev_s, dev_ll_protanopia, dev_mm_protanopia, dev_ss_protanopia, size);
    correct_protanopia_kernel<<<blocksPerGrid, threadsPerBlock, 0, streamProtanopia>>>(dev_l, dev_m, dev_s, dev_ll_corrected_protanopia, dev_mm_corrected_protanopia, dev_ss_corrected_protanopia, size);

    // Process 2: simulate_deuteranopia_kernel in streamDeuteranopia
    simulate_deuteranopia_kernel<<<blocksPerGrid, threadsPerBlock, 0, streamDeuteranopia>>>(dev_l, dev_m, dev_s, dev_ll_deuteranopia, dev_mm_deuteranopia, dev_ss_deuteranopia, size);
    correct_deuteranopia_kernel<<<blocksPerGrid, threadsPerBlock, 0, streamDeuteranopia>>>(dev_l, dev_m, dev_s, dev_ll_corrected_deuteranopia, dev_mm_corrected_deuteranopia, dev_ss_corrected_deuteranopia, size);

    // Process 3: simulate_tritanopia_kernel in streamTritanopia
    simulate_tritanopia_kernel<<<blocksPerGrid, threadsPerBlock, 0, streamTritanopia>>>(dev_l, dev_m, dev_s, dev_ll_tritanopia, dev_mm_tritanopia, dev_ss_tritanopia, size);
    correct_tritanopia_kernel<<<blocksPerGrid, threadsPerBlock, 0, streamTritanopia>>>(dev_l, dev_m, dev_s, dev_ll_corrected_tritanopia, dev_mm_corrected_tritanopia, dev_ss_corrected_tritanopia, size);

    // Allocate host buffers for the output
    unsigned char *protanopiaBuffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char));
    unsigned char *deuteranopiaBuffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char));
    unsigned char *tritanopiaBuffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char));
    // Convert corrected LMS back to RGB and write to file
    unsigned char *corrected_protanopiaBuffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char));
    unsigned char *corrected_deuteranopiaBuffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char));
    unsigned char *corrected_tritanopiaBuffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char));

    // Copy results back to host in streamProtanopia
    cudaMemcpyAsync(l_protanopia, dev_ll_protanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamProtanopia);
    cudaMemcpyAsync(m_protanopia, dev_mm_protanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamProtanopia);
    cudaMemcpyAsync(s_protanopia, dev_ss_protanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamProtanopia);
    cudaMemcpyAsync(l_corrected_protanopia, dev_ll_corrected_protanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamProtanopia);
    cudaMemcpyAsync(m_corrected_protanopia, dev_mm_corrected_protanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamProtanopia);
    cudaMemcpyAsync(s_corrected_protanopia, dev_ss_corrected_protanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamProtanopia);

    // Copy results back to host in streamDeuteranopia
    cudaMemcpyAsync(l_deuteranopia, dev_ll_deuteranopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamDeuteranopia);
    cudaMemcpyAsync(m_deuteranopia, dev_mm_deuteranopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamDeuteranopia);
    cudaMemcpyAsync(s_deuteranopia, dev_ss_deuteranopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamDeuteranopia);
    cudaMemcpyAsync(l_corrected_deuteranopia, dev_ll_corrected_deuteranopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamDeuteranopia);
    cudaMemcpyAsync(m_corrected_deuteranopia, dev_mm_corrected_deuteranopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamDeuteranopia);
    cudaMemcpyAsync(s_corrected_deuteranopia, dev_ss_corrected_deuteranopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamDeuteranopia);

    // Copy results back to host in streamTritanopia
    cudaMemcpyAsync(l_tritanopia, dev_ll_tritanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamTritanopia);
    cudaMemcpyAsync(m_tritanopia, dev_mm_tritanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamTritanopia);
    cudaMemcpyAsync(s_tritanopia, dev_ss_tritanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamTritanopia);
    cudaMemcpyAsync(l_corrected_protanopia, dev_ll_corrected_tritanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamTritanopia);
    cudaMemcpyAsync(m_corrected_protanopia, dev_mm_corrected_tritanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamTritanopia);
    cudaMemcpyAsync(s_corrected_protanopia, dev_ss_corrected_tritanopia, size * sizeof(float), cudaMemcpyDeviceToHost, streamTritanopia);

    // Wait for streamProtanopia to finish
    cudaStreamSynchronize(streamProtanopia);
    // Wait for streamDeuteranopia to finish
    cudaStreamSynchronize(streamDeuteranopia);
    // Wait for streamTritanopia to finish
    cudaStreamSynchronize(streamTritanopia);

    // Convert LMS back to RGB for protanopia and write to file
    for (int i = 0; i < size; i++)
    {
        unsigned char r, g, b;
        lms2rgb(l_protanopia[i], m_protanopia[i], s_protanopia[i], &r, &g, &b);
        protanopiaBuffer[i * 3] = r;
        protanopiaBuffer[i * 3 + 1] = g;
        protanopiaBuffer[i * 3 + 2] = b;

        lms2rgb(l_deuteranopia[i], m_deuteranopia[i], s_deuteranopia[i], &r, &g, &b);
        deuteranopiaBuffer[i * 3] = r;
        deuteranopiaBuffer[i * 3 + 1] = g;
        deuteranopiaBuffer[i * 3 + 2] = b;

        lms2rgb(l_tritanopia[i], m_tritanopia[i], s_tritanopia[i], &r, &g, &b);
        tritanopiaBuffer[i * 3] = r;
        tritanopiaBuffer[i * 3 + 1] = g;
        tritanopiaBuffer[i * 3 + 2] = b;

        lms2rgb(l_corrected_protanopia[i], m_corrected_protanopia[i], s_corrected_protanopia[i], &r, &g, &b);
        corrected_protanopiaBuffer[i * 3] = r;
        corrected_protanopiaBuffer[i * 3 + 1] = g;
        corrected_protanopiaBuffer[i * 3 + 2] = b;

        lms2rgb(l_corrected_deuteranopia[i], m_corrected_deuteranopia[i], s_corrected_deuteranopia[i], &r, &g, &b);
        corrected_deuteranopiaBuffer[i * 3] = r;
        corrected_deuteranopiaBuffer[i * 3 + 1] = g;
        corrected_deuteranopiaBuffer[i * 3 + 2] = b;

        lms2rgb(l_corrected_tritanopia[i], m_corrected_tritanopia[i], s_corrected_tritanopia[i], &r, &g, &b);
        corrected_tritanopiaBuffer[i * 3] = r;
        corrected_tritanopiaBuffer[i * 3 + 1] = g;
        corrected_tritanopiaBuffer[i * 3 + 2] = b;
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

    free(protanopiaBuffer);
    free(deuteranopiaBuffer);
    free(tritanopiaBuffer);

    free(corrected_protanopiaBuffer);
    free(corrected_deuteranopiaBuffer);
    free(corrected_tritanopiaBuffer);

    free(buffer);

    free(l_protanopia);
    free(m_protanopia);
    free(s_protanopia);

    free(l_deuteranopia);
    free(m_deuteranopia);
    free(s_deuteranopia);

    free(l_tritanopia);
    free(m_tritanopia);
    free(s_tritanopia);

    free(l_corrected_protanopia);
    free(m_corrected_protanopia);
    free(s_corrected_protanopia);

    free(l_corrected_deuteranopia);
    free(m_corrected_deuteranopia);
    free(s_corrected_deuteranopia);

    free(l_corrected_tritanopia);
    free(m_corrected_tritanopia);
    free(s_corrected_tritanopia);

    cudaFree(dev_l);
    cudaFree(dev_m);
    cudaFree(dev_s);
    
    cudaFree(dev_ll_protanopia);
    cudaFree(dev_mm_protanopia);
    cudaFree(dev_ss_protanopia);

    cudaFree(dev_ll_deuteranopia);
    cudaFree(dev_mm_deuteranopia);
    cudaFree(dev_ss_deuteranopia);

    cudaFree(dev_ll_tritanopia);
    cudaFree(dev_mm_tritanopia);
    cudaFree(dev_ss_tritanopia);

    cudaFree(dev_ll_corrected_protanopia);
    cudaFree(dev_mm_corrected_protanopia);
    cudaFree(dev_ss_corrected_protanopia);

    cudaFree(dev_ll_corrected_deuteranopia);
    cudaFree(dev_mm_corrected_deuteranopia);
    cudaFree(dev_ss_corrected_deuteranopia);

    cudaFree(dev_ll_corrected_tritanopia);
    cudaFree(dev_mm_corrected_tritanopia);
    cudaFree(dev_ss_corrected_tritanopia);

}

int main(int argc, char *argv[])
{
    processImageWithCUDA("lena_color", "temp");
}