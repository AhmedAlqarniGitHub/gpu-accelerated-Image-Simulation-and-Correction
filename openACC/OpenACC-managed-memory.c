#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <openacc.h>

void rgb2lms(unsigned char r, unsigned char g, unsigned char b,
             float *l, float *m, float *s)
{

    float rr = (r / 255.0);
    float gg = (g / 255.0);
    float bb = (b / 255.0);

    // convert to srgb
    rr = rr <= 0.04045 ? (rr) / 12.92 : pow((rr + 0.055) / 1.055, 2.4);
    gg = gg <= 0.04045 ? (gg) / 12.92 : pow((gg + 0.055) / 1.055, 2.4);
    bb = bb <= 0.04045 ? (bb) / 12.92 : pow((bb + 0.055) / 1.055, 2.4);

    // convert to lms
    *l = (rr * 0.31399022) + (gg * 0.15537241) + (bb * 0.01775239);
    *m = (rr * 0.63951294) + (gg * 0.75789446) + (bb * 0.10944209);
    *s = (rr * 0.04649755) + (gg * 0.08670142) + (bb * 0.87256922);
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

void simulate_protanopia(float l, float m, float s, float *ll, float *mm, float *ss)
{
    *ll = (0.0 * l) + (2.02344 * m) + (-2.52581 * s);
    *mm = (0.0 * l) + (1.0 * m) + (0.0 * s);
    *ss = (0.0 * l) + (0.0 * m) + (1.0 * s);

    return;
}
void simulate_deuteranopia(float l, float m, float s, float *ll, float *mm, float *ss)
{
    *ll = (1.0 * l) + (0.0 * m) + (0.0 * s);
    *mm = (0.494207 * l) + (0.0 * m) + (1.24827 * s);
    *ss = (0.0 * l) + (0.0 * m) + (1.0 * s);

    return;
}
void simulate_tritanopia(float l, float m, float s, float *ll, float *mm, float *ss)
{
    *ll = (1.0 * l) + (0.0 * m) + (0.0 * s);
    *mm = (0.0 * l) + (1.0 * m) + (0.0 * s);
    *ss = (-0.395913 * l) + (0.801109 * m) + (0.0 * s);

    return;
}

void correct_protanopia(float l, float m, float s, float *ll, float *mm, float *ss)
{
    *ll = (0.0 * l) + (2.02344 * m) + (-2.52581 * s);
    *mm = (0.0 * l) + (1.0 * m) + (0.0 * s);
    *ss = (0.0 * l) + (0.0 * m) + (1.0 * s);
}
void correct_deuteranopia(float l, float m, float s, float *ll, float *mm, float *ss)
{
    *ll = (1.0 * l) + (0.0 * m) + (0.0 * s);
    *mm = (0.494207 * l) + (0.0 * m) + (1.24827 * s);
    *ss = (0.0 * l) + (0.0 * m) + (1.0 * s);
}
void correct_tritanopia(float l, float m, float s, float *ll, float *mm, float *ss)
{
    *ll = (1.0 * l) + (0.0 * m) + (0.0 * s);
    *mm = (0.0 * l) + (1.0 * m) + (0.0 * s);
    *ss = (-0.395913 * l) + (0.801109 * m) + (0.0 * s);
}

int simulate_cvd_protanopia(char imageFileName[100], unsigned char header[54], int size, unsigned char buffer[size][3], int bitDepth, unsigned char colorTable[1024])
{
    char ImageFilePath[100];
    sprintf(ImageFilePath, "out/%s/simulate_protanopia.bmp", imageFileName);
    FILE *fOut = fopen(ImageFilePath, "w+"); // Output File name

    if (fOut == NULL) // check if the input file has not been opened succesfully.
    {
        printf("File did not open.\n");
    }

    int i, j, y, x;
    float l, m, s; // original
    unsigned char r, g, b;
    float ll, mm, ss; // updated

    fwrite(header, sizeof(unsigned char), 54, fOut); // write the header back

    unsigned char out[size][3]; // store the output image data

#pragma acc parallel loop
    for (i = 0; i < size; i++)
    {
        b = buffer[i][0]; // blue
        g = buffer[i][1]; // green
        r = buffer[i][2]; // red

        rgb2lms(r, g, b, &l, &m, &s);

        simulate_protanopia(l, m, s, &ll, &mm, &ss);

        lms2rgb(ll, mm, ss, &r, &g, &b);

        out[i][0] = r;
        out[i][1] = g;
        out[i][2] = b;
    }

    for (i = 0; i < size; i++)
    {
        putc(out[i][2], fOut);
        putc(out[i][1], fOut);
        putc(out[i][0], fOut);
    }

    fclose(fOut);
    return 0;
}

int simulate_cvd_deuteranopia(char imageFileName[100], unsigned char header[54], int size, unsigned char buffer[size][3], int bitDepth, unsigned char colorTable[1024])
{
    char ImageFilePath[150];
    sprintf(ImageFilePath, "out/%s/simulate_deuteranopia.bmp", imageFileName);
    FILE *fOut = fopen(ImageFilePath, "w+"); // Output File name

    int i, j, y, x;
    float l, m, s; // original
    unsigned char r, g, b;
    float ll, mm, ss; // updated

    fwrite(header, sizeof(unsigned char), 54, fOut); // write the header back
    unsigned char out[size][3];                      // store the output image data

#pragma acc parallel loop
    for (i = 0; i < size; i++)
    {
        b = buffer[i][0]; // blue
        g = buffer[i][1]; // green
        r = buffer[i][2]; // red

        rgb2lms(r, g, b, &l, &m, &s);

        simulate_deuteranopia(l, m, s, &ll, &mm, &ss);

        lms2rgb(ll, mm, ss, &r, &g, &b);

        out[i][0] = r;
        out[i][1] = g;
        out[i][2] = b;
    }

    for (i = 0; i < size; i++)
    {
        putc(out[i][2], fOut);
        putc(out[i][1], fOut);
        putc(out[i][0], fOut);
    }

    fclose(fOut);
    return 0;
}

int simulate_cvd_tritanopia(char imageFileName[100], unsigned char header[54], int size, unsigned char buffer[size][3], int bitDepth, unsigned char colorTable[1024])
{
    char ImageFilePath[150];
    sprintf(ImageFilePath, "out/%s/simulate_tritanopia.bmp", imageFileName);
    FILE *fOut = fopen(ImageFilePath, "w+"); // Output File name

    int i, j, y, x;
    float l, m, s; // original
    unsigned char r, g, b;
    float ll, mm, ss; // updated

    fwrite(header, sizeof(unsigned char), 54, fOut); // write the header back
    unsigned char out[size][3];                      // store the output image data

#pragma acc parallel loop
    for (i = 0; i < size; i++)
    {
        b = buffer[i][0]; // blue
        g = buffer[i][1]; // green
        r = buffer[i][2]; // red

        rgb2lms(r, g, b, &l, &m, &s);

        simulate_tritanopia(l, m, s, &ll, &mm, &ss);

        lms2rgb(ll, mm, ss, &r, &g, &b);

        out[i][0] = r;
        out[i][1] = g;
        out[i][2] = b;
    }

    for (i = 0; i < size; i++)
    {
        putc(out[i][2], fOut);
        putc(out[i][1], fOut);
        putc(out[i][0], fOut);
    }

    fclose(fOut);

    return 0;
}

int correct_cvd_protanopia(char imageFileName[100], unsigned char header[54], int size, unsigned char buffer[size][3], int bitDepth, unsigned char colorTable[1024])
{

    char ImageFilePath[150];
    sprintf(ImageFilePath, "out/%s/correct_protanopia.bmp", imageFileName);
    FILE *fOut = fopen(ImageFilePath, "w+"); // Output File name

    fwrite(header, sizeof(unsigned char), 54, fOut); // write the header back
    unsigned char out[size][3];                      // store the output image data
    unsigned int index;

    unsigned char r, g, b;
    float rr, gg, bb;
    float l, m, s;    // original
    float ll, mm, ss; // updated

#pragma acc parallel loop
    for (index = 0; index < size; index++)
    {
        r = buffer[index][0]; // red
        g = buffer[index][1]; // green
        b = buffer[index][2]; // blue

        // convert to lms
        l = (17.8824 * r) + (43.5161 * g) + (4.11935 * b);    // long
        m = (3.45565 * r) + (27.1554 * g) + (3.86714 * b);    // medium
        s = (0.0299566 * r) + (0.184309 * g) + (1.46709 * b); // short

        correct_protanopia(l, m, s, &ll, &mm, &ss);

        // convert back to rgb
        rr = (0.0809444479 * ll) + (-0.130504409 * mm) + (0.116721066 * ss);
        gg = (-0.0102485335 * ll) + (0.0540193266 * mm) + (-0.113614708 * ss);
        bb = (-0.000365296938 * ll) + (-0.00412161469 * mm) + (0.693511405 * ss);

        // calculate error
        rr = r - rr;
        gg = g - gg;
        bb = b - bb;

        // shift the color towards visible spectrum and add compensation
        rr = r;
        gg = ((0.7 * rr) + (1.0 * gg)) + g;
        bb = ((0.7 * rr) + (1.0 * bb)) + b;

        // clamp values towards unsigned char
        r = (rr > 255) ? 255 : ((rr < 0) ? 0 : rr);
        g = (gg > 255) ? 255 : ((gg < 0) ? 0 : gg);
        b = (bb > 255) ? 255 : ((bb < 0) ? 0 : bb);

        out[index][0] = r;
        out[index][1] = g;
        out[index][2] = b;
    }

    for (index = 0; index < size; index++)
    {
        putc(out[index][2], fOut);
        putc(out[index][1], fOut);
        putc(out[index][0], fOut);
    }

    fclose(fOut);
    return 0;
}

int correct_cvd_deuteranopia(char imageFileName[100], unsigned char header[54], int size, unsigned char buffer[size][3], int bitDepth, unsigned char colorTable[1024])
{

    char ImageFilePath[150];
    sprintf(ImageFilePath, "out/%s/correct_deuteranopia.bmp", imageFileName);
    FILE *fOut = fopen(ImageFilePath, "w+"); // Output File name

    fwrite(header, sizeof(unsigned char), 54, fOut); // write the header back
    unsigned char out[size][3];                      // store the output image data
    unsigned int index;

    unsigned char r, g, b;
    float rr, gg, bb;
    float l, m, s;    // original
    float ll, mm, ss; // updated

#pragma acc parallel loop
    for (index = 0; index < size; index++)
    {
        r = buffer[index][0]; // red
        g = buffer[index][1]; // green
        b = buffer[index][2]; // blue

        // convert to lms
        l = (17.8824 * r) + (43.5161 * g) + (4.11935 * b);    // long
        m = (3.45565 * r) + (27.1554 * g) + (3.86714 * b);    // medium
        s = (0.0299566 * r) + (0.184309 * g) + (1.46709 * b); // short

        correct_deuteranopia(l, m, s, &ll, &mm, &ss);

        // convert back to rgb
        rr = (0.0809444479 * ll) + (-0.130504409 * mm) + (0.116721066 * ss);
        gg = (-0.0102485335 * ll) + (0.0540193266 * mm) + (-0.113614708 * ss);
        bb = (-0.000365296938 * ll) + (-0.00412161469 * mm) + (0.693511405 * ss);

        // calculate error
        rr = r - rr;
        gg = g - gg;
        bb = b - bb;

        // shift the color towards visible spectrum and add compensation
        rr = r;
        gg = ((0.7 * rr) + (1.0 * gg)) + g;
        bb = ((0.7 * rr) + (1.0 * bb)) + b;

        // clamp values towards unsigned char
        r = (rr > 255) ? 255 : ((rr < 0) ? 0 : rr);
        g = (gg > 255) ? 255 : ((gg < 0) ? 0 : gg);
        b = (bb > 255) ? 255 : ((bb < 0) ? 0 : bb);

        out[index][0] = r;
        out[index][1] = g;
        out[index][2] = b;
    }

    for (index = 0; index < size; index++)
    {
        putc(out[index][2], fOut);
        putc(out[index][1], fOut);
        putc(out[index][0], fOut);
    }

    fclose(fOut);
    return 0;
}
int correct_cvd_tritanopia(char imageFileName[100], unsigned char header[54], int size, unsigned char buffer[size][3], int bitDepth, unsigned char colorTable[1024])
{

    char ImageFilePath[150];
    sprintf(ImageFilePath, "out/%s/correct_tritanopia.bmp", imageFileName);
    FILE *fOut = fopen(ImageFilePath, "w+"); // Output File name

    fwrite(header, sizeof(unsigned char), 54, fOut); // write the header back
    unsigned char out[size][3];                      // store the output image data
    unsigned int index;

    unsigned char r, g, b;
    float rr, gg, bb;
    float l, m, s;    // original
    float ll, mm, ss; // updated

#pragma acc parallel loop
    for (index = 0; index < size; index++)
    {
        r = buffer[index][0]; // red
        g = buffer[index][1]; // green
        b = buffer[index][2]; // blue

        // convert to lms
        l = (17.8824 * r) + (43.5161 * g) + (4.11935 * b);    // long
        m = (3.45565 * r) + (27.1554 * g) + (3.86714 * b);    // medium
        s = (0.0299566 * r) + (0.184309 * g) + (1.46709 * b); // short

        correct_tritanopia(l, m, s, &ll, &mm, &ss);

        // convert back to rgb
        rr = (0.0809444479 * ll) + (-0.130504409 * mm) + (0.116721066 * ss);
        gg = (-0.0102485335 * ll) + (0.0540193266 * mm) + (-0.113614708 * ss);
        bb = (-0.000365296938 * ll) + (-0.00412161469 * mm) + (0.693511405 * ss);

        // calculate error
        rr = r - rr;
        gg = g - gg;
        bb = b - bb;

        // shift the color towards visible spectrum and add compensation
        rr = r;
        gg = ((0.7 * rr) + (1.0 * gg)) + g;
        bb = ((0.7 * rr) + (1.0 * bb)) + b;

        // clamp values towards unsigned char
        r = (rr > 255) ? 255 : ((rr < 0) ? 0 : rr);
        g = (gg > 255) ? 255 : ((gg < 0) ? 0 : gg);
        b = (bb > 255) ? 255 : ((bb < 0) ? 0 : bb);

        out[index][0] = r;
        out[index][1] = g;
        out[index][2] = b;
    }

    // write image data back to the file
    for (index = 0; index < size; index++)
    {
        putc(out[index][2], fOut);
        putc(out[index][1], fOut);
        putc(out[index][0], fOut);
    }

    fclose(fOut);
    return 0;
}

int driver(char imageFileName[])
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

    if (bitDepth <= 8) // if ColorTable present, extract it.
    {
        fread(colorTable, sizeof(unsigned char), 1024, fIn);
    }

    int size = height * width;     // calculate image size
    unsigned char buffer[size][3]; // to store the image data

    for (i = 0; i < size; i++)
    {

        buffer[i][2] = getc(fIn); // blue
        buffer[i][1] = getc(fIn); // green
        buffer[i][0] = getc(fIn); // red
    }

    printf("height: %d\n", height);
    printf("width: %d\n", width);
    printf("size: %d\n", size);

    for (int i = 0; i < 100; i++)
    {
        simulate_cvd_protanopia(imageFileName, header, size, buffer, bitDepth, colorTable);
        simulate_cvd_deuteranopia(imageFileName, header, size, buffer, bitDepth, colorTable);
        simulate_cvd_tritanopia(imageFileName, header, size, buffer, bitDepth, colorTable);

        correct_cvd_protanopia(imageFileName, header, size, buffer, bitDepth, colorTable);
        correct_cvd_deuteranopia(imageFileName, header, size, buffer, bitDepth, colorTable);
        correct_cvd_tritanopia(imageFileName, header, size, buffer, bitDepth, colorTable);
    }
}

int main(int argc, char *argv[])
{
    if (acc_get_device_type() == acc_device_nvidia)
    {
        printf("Running on NVIDIA GPU with %d devices available.\n", acc_get_num_devices(acc_device_nvidia));
    }
    else
    {
        printf("Not running on an NVIDIA GPU.\n");
    }
    driver("lena_color");
}