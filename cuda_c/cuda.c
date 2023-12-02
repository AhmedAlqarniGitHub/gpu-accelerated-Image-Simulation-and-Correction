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
        l[i] = (rr * 0.31399022) + (gg * 0.15537241) + (bb * 0.01775239);
        m[i] = (rr * 0.63951294) + (gg * 0.75789446) + (bb * 0.10944209);
        s[i] = (rr * 0.04649755) + (gg * 0.08670142) + (bb * 0.87256922);
    }
    
    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;


    // Allocate device memory
    float *dev_l, *dev_m, *dev_s, *dev_ll, *dev_mm, *dev_ss;
    cudaMalloc((void **)&dev_l, size * sizeof(float));
    cudaMalloc((void **)&dev_m, size * sizeof(float));
    cudaMalloc((void **)&dev_s, size * sizeof(float));
    cudaMalloc((void **)&dev_ll, size * sizeof(float));
    cudaMalloc((void **)&dev_mm, size * sizeof(float));
    cudaMalloc((void **)&dev_ss, size * sizeof(float));

    // Copy data from host to device in both streams
    cudaMemcpyAsync(dev_l, l, size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(dev_m, m, size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(dev_s, s, size * sizeof(float), cudaMemcpyHostToDevice, stream1);

    cudaMemcpyAsync(dev_l, l, size * sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(dev_m, m, size * sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(dev_s, s, size * sizeof(float), cudaMemcpyHostToDevice, stream2);

    // Process 1: simulate_protanopia_kernel in stream1
    simulate_protanopia_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(dev_l, dev_m, dev_s, dev_ll, dev_mm, dev_ss, size);

    // Process 2: simulate_deuteranopia_kernel in stream2
    simulate_deuteranopia_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(dev_l, dev_m, dev_s, dev_ll, dev_mm, dev_ss, size);

    // Allocate host buffers for the output
    unsigned char *protanopiaBuffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char));
    unsigned char *deuteranopiaBuffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char));

    // Copy results back to host in stream1
    cudaMemcpyAsync(l, dev_ll, size * sizeof(float), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(m, dev_mm, size * sizeof(float), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(s, dev_ss, size * sizeof(float), cudaMemcpyDeviceToHost, stream1);

    // Copy results back to host in stream2
    cudaMemcpyAsync(l, dev_ll, size * sizeof(float), cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(m, dev_mm, size * sizeof(float), cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(s, dev_ss, size * sizeof(float), cudaMemcpyDeviceToHost, stream2);

    // Wait for stream1 to finish
    cudaStreamSynchronize(stream1);

    // Convert LMS back to RGB for protanopia and write to file
    for (int i = 0; i < size; i++)
    {
        unsigned char r, g, b;
        lms2rgb(l[i], m[i], s[i], &r, &g, &b);
        protanopiaBuffer[i * 3] = r;
        protanopiaBuffer[i * 3 + 1] = g;
        protanopiaBuffer[i * 3 + 2] = b;
    }

    char protanopiaFileName[150];
    sprintf(protanopiaFileName, "%s_protanopia.bmp", outputFileName);
    FILE *fOut = fopen(protanopiaFileName, "wb");
    fwrite(header, sizeof(unsigned char), 54, fOut);
    fwrite(protanopiaBuffer, sizeof(unsigned char), size * 3, fOut);
    fclose(fOut);

    // Wait for stream2 to finish
    cudaStreamSynchronize(stream2);

    // Convert LMS back to RGB for deuteranopia and write to file
    for (int i = 0; i < size; i++)
    {
        unsigned char r, g, b;
        lms2rgb(l[i], m[i], s[i], &r, &g, &b);
        deuteranopiaBuffer[i * 3] = r;
        deuteranopiaBuffer[i * 3 + 1] = g;
        deuteranopiaBuffer[i * 3 + 2] = b;
    }

    char deuteranopiaFileName[150];
    sprintf(deuteranopiaFileName, "%s_deuteranopia.bmp", outputFileName);
    FILE *fOut2 = fopen(deuteranopiaFileName, "wb");
    fwrite(header, sizeof(unsigned char), 54, fOut2);
    fwrite(deuteranopiaBuffer, sizeof(unsigned char), size * 3, fOut2);
    fclose(fOut2);

    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    free(protanopiaBuffer);
    free(deuteranopiaBuffer);
    free(buffer);
    free(l);
    free(m);
    free(s);
    cudaFree(dev_l);
    cudaFree(dev_m);
    cudaFree(dev_s);
    cudaFree(dev_ll);
    cudaFree(dev_mm);
    cudaFree(dev_ss);
}

int main(int argc, char *argv[])
{
    processImageWithCUDA("lena_color", "temp");
}#include <stdio.h>
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
        l[i] = (rr * 0.31399022) + (gg * 0.15537241) + (bb * 0.01775239);
        m[i] = (rr * 0.63951294) + (gg * 0.75789446) + (bb * 0.10944209);
        s[i] = (rr * 0.04649755) + (gg * 0.08670142) + (bb * 0.87256922);
    }
    
    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;


    // Allocate device memory
    float *dev_l, *dev_m, *dev_s, *dev_ll, *dev_mm, *dev_ss;
    cudaMalloc((void **)&dev_l, size * sizeof(float));
    cudaMalloc((void **)&dev_m, size * sizeof(float));
    cudaMalloc((void **)&dev_s, size * sizeof(float));
    cudaMalloc((void **)&dev_ll, size * sizeof(float));
    cudaMalloc((void **)&dev_mm, size * sizeof(float));
    cudaMalloc((void **)&dev_ss, size * sizeof(float));

    // Copy data from host to device in both streams
    cudaMemcpyAsync(dev_l, l, size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(dev_m, m, size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(dev_s, s, size * sizeof(float), cudaMemcpyHostToDevice, stream1);

    cudaMemcpyAsync(dev_l, l, size * sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(dev_m, m, size * sizeof(float), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(dev_s, s, size * sizeof(float), cudaMemcpyHostToDevice, stream2);

    // Process 1: simulate_protanopia_kernel in stream1
    simulate_protanopia_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(dev_l, dev_m, dev_s, dev_ll, dev_mm, dev_ss, size);

    // Process 2: simulate_deuteranopia_kernel in stream2
    simulate_deuteranopia_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(dev_l, dev_m, dev_s, dev_ll, dev_mm, dev_ss, size);

    // Allocate host buffers for the output
    unsigned char *protanopiaBuffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char));
    unsigned char *deuteranopiaBuffer = (unsigned char *)malloc(size * 3 * sizeof(unsigned char));

    // Copy results back to host in stream1
    cudaMemcpyAsync(l, dev_ll, size * sizeof(float), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(m, dev_mm, size * sizeof(float), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(s, dev_ss, size * sizeof(float), cudaMemcpyDeviceToHost, stream1);

    // Copy results back to host in stream2
    cudaMemcpyAsync(l, dev_ll, size * sizeof(float), cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(m, dev_mm, size * sizeof(float), cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(s, dev_ss, size * sizeof(float), cudaMemcpyDeviceToHost, stream2);

    // Wait for stream1 to finish
    cudaStreamSynchronize(stream1);

    // Convert LMS back to RGB for protanopia and write to file
    for (int i = 0; i < size; i++)
    {
        unsigned char r, g, b;
        lms2rgb(l[i], m[i], s[i], &r, &g, &b);
        protanopiaBuffer[i * 3] = r;
        protanopiaBuffer[i * 3 + 1] = g;
        protanopiaBuffer[i * 3 + 2] = b;
    }

    char protanopiaFileName[150];
    sprintf(protanopiaFileName, "%s_protanopia.bmp", outputFileName);
    FILE *fOut = fopen(protanopiaFileName, "wb");
    fwrite(header, sizeof(unsigned char), 54, fOut);
    fwrite(protanopiaBuffer, sizeof(unsigned char), size * 3, fOut);
    fclose(fOut);

    // Wait for stream2 to finish
    cudaStreamSynchronize(stream2);

    // Convert LMS back to RGB for deuteranopia and write to file
    for (int i = 0; i < size; i++)
    {
        unsigned char r, g, b;
        lms2rgb(l[i], m[i], s[i], &r, &g, &b);
        deuteranopiaBuffer[i * 3] = r;
        deuteranopiaBuffer[i * 3 + 1] = g;
        deuteranopiaBuffer[i * 3 + 2] = b;
    }

    char deuteranopiaFileName[150];
    sprintf(deuteranopiaFileName, "%s_deuteranopia.bmp", outputFileName);
    FILE *fOut2 = fopen(deuteranopiaFileName, "wb");
    fwrite(header, sizeof(unsigned char), 54, fOut2);
    fwrite(deuteranopiaBuffer, sizeof(unsigned char), size * 3, fOut2);
    fclose(fOut2);

    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    free(protanopiaBuffer);
    free(deuteranopiaBuffer);
    free(buffer);
    free(l);
    free(m);
    free(s);
    cudaFree(dev_l);
    cudaFree(dev_m);
    cudaFree(dev_s);
    cudaFree(dev_ll);
    cudaFree(dev_mm);
    cudaFree(dev_ss);
}

int main(int argc, char *argv[])
{
    processImageWithCUDA("lena_color", "temp");
}