#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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


void simulate_color_vision_deficiency(float l, float m, float s, float *ll, float *mm, float *ss, int type)
{
    // type: 1 for protanopia, 2 for deuteranopia, 3 for tritanopia
    switch (type)
    {
    case 1: // Protanopia (L cones are defective)
        *ll = 0.0 * l + 2.02344 * m - 2.52581 * s;
        *mm = m;
        *ss = s;
        break;
    case 2: // Deuteranopia (M cones are defective)
        *ll = l;
        *mm = 0.494207 * l + 1.24827 * s;
        *ss = s;
        break;
    case 3: // Tritanopia (S cones are defective)
        *ll = l;
        *mm = m;
        *ss = -0.395913 * l + 0.801109 * m;
        break;
    default:
        // If type is not recognized, do not modify the LMS values
        *ll = l;
        *mm = m;
        *ss = s;
    }
}


int process_image(char *inputFileName, char *outputFileName, int cvdType)
{
    FILE *fIn = fopen(inputFileName, "rb");
    FILE *fOut = fopen(outputFileName, "wb");
    unsigned char header[54];
    unsigned char colorTable[1024];
    int height, width, bitDepth;

    if (fIn == NULL || fOut == NULL)
    {
        printf("File error.\n");
        return -1;
    }

    fread(header, sizeof(unsigned char), 54, fIn); // read the 54-byte header
    fwrite(header, sizeof(unsigned char), 54, fOut); // write the header to output file

    height = *(int *)&header[18];
    width = *(int *)&header[22];
    bitDepth = *(int *)&header[28];

    if (bitDepth <= 8) // if ColorTable present, read it.
    {
        fread(colorTable, sizeof(unsigned char), 1024, fIn);
        fwrite(colorTable, sizeof(unsigned char), 1024, fOut);
    }

    int imgSize = height * width;
    unsigned char buffer[imgSize][3]; // to store the image data

    for (int i = 0; i < imgSize; i++)
    {
        fread(buffer[i], sizeof(unsigned char), 3, fIn); // read RGB values

        unsigned char r = buffer[i][2];
        unsigned char g = buffer[i][1];
        unsigned char b = buffer[i][0];

        float l, m, s, ll, mm, ss;
        rgb2lms(r, g, b, &l, &m, &s);
        simulate_color_vision_deficiency(l, m, s, &ll, &mm, &ss, cvdType);
        lms2rgb(ll, mm, ss, &r, &g, &b);

        buffer[i][2] = r;
        buffer[i][1] = g;
        buffer[i][0] = b;

        fwrite(buffer[i], sizeof(unsigned char), 3, fOut); // write processed RGB values
    }

    fclose(fIn);
    fclose(fOut);
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("Usage: program_name input.bmp output.bmp cvdType\n");
        printf("cvdType: 1 for Protanopia, 2 for Deuteranopia, 3 for Tritanopia\n");
        return -1;
    }

    char *inputFileName = argv[1];
    char *outputFileName = argv[2];
    int cvdType = atoi(argv[3]);

    process_image(inputFileName, outputFileName, cvdType);
    return 0;
}
