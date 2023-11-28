#include <stdio.h>
#include <time.h>
#include <omp.h>

int image_rgbtogray(char imageFileName[100], unsigned char header[54], int height, int width, unsigned char buffer[width][height][3], unsigned char colorTable[1024])
{
	char ImageFilePath[150];
	sprintf(ImageFilePath, "out/%s/image_gray.bmp", imageFileName);

	FILE *fOut = fopen(ImageFilePath, "w+"); // Output File name

	int i, j;
	int tempH, tempW;
	unsigned char heightA[4];
	unsigned char widthA[4];
	int bitDepth = *(int *)&header[28];
	int size = height * width; // calculate image size

	double out_buffer[width][height];

	fwrite(header, sizeof(unsigned char), 54, fOut); // write the header back
	if (bitDepth <= 8)								 // if ColorTable present, extract it.
	{
		fwrite(colorTable, sizeof(unsigned char), 1024, fOut);
	}
	
#pragma omp parallel for schedule(guided, 10) private(j)
	for (i = 0; i < width; i++)		// to rotate right
	{
		for (j = 0; j < height; j++)
		{
			out_buffer[i][j] = (buffer[i][j][0] * 0.3 + buffer[i][j][1] * 0.59 + buffer[i][j][2] * 0.11);
		}
	}

#pragma omp parallel for num_threads(1)
	for (i = 0; i < width; i++)
	{
		for (j = 0; j < height; j++)
		{
			putc(out_buffer[i][j], fOut);
			putc(out_buffer[i][j], fOut);
			putc(out_buffer[i][j], fOut);
		}
	}
	fclose(fOut);
	return 0;
}
