#include <stdio.h>
#include <time.h>
#include "color_blindess.c"

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
	unsigned char out[size][3];						 // store the output image data

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

#pragma omp parallel for num_threads(1)
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
	unsigned char out[size][3];						 // store the output image data

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

#pragma omp parallel for num_threads(1)
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
	unsigned char out[size][3];						 // store the output image data

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

#pragma omp parallel for num_threads(1)
	for (i = 0; i < size; i++)
	{
		putc(out[i][2], fOut);
		putc(out[i][1], fOut);
		putc(out[i][0], fOut);
	}

	fclose(fOut);

	return 0;
}
