#include <stdio.h>
#define THRESHOLD 40

int image_dark(char imageFileName[100], unsigned char *header, unsigned char *colorTable, int size, unsigned char *buffer)
{
	// FILE *fOut = fopen("out/lena_dark.bmp", "w+"); // Output File name
	char ImageFilePath[150];
	sprintf(ImageFilePath, "out/%s/image_dark.bmp", imageFileName);
	FILE *fOut = fopen(ImageFilePath, "w+"); // Output File name

	int i;

	fwrite(header, sizeof(unsigned char), 54, fOut); // write the header back

	// extract image height, width and bitDepth from imageHeader
	int bitDepth = *(int *)&header[28];

	if (bitDepth <= 8) // if ColorTable present, extract it.
	{
		fwrite(colorTable, sizeof(unsigned char), 1024, fOut);
	}

	unsigned char image_buffer[size];
	// Parallelizable segment
#pragma omp parallel for schedule(guided, 10)
	for (i = 0; i < size; i++) // decreasing pixel values to get image bright
	{
		if (buffer[i] > THRESHOLD)
			image_buffer[i] = buffer[i] - THRESHOLD;
		else
			image_buffer[i] = 0;
		// image_buffer[i] = buffer[i];
	}

	fwrite(image_buffer, sizeof(unsigned char), size, fOut); // write back to the output image
	fclose(fOut);

	return 0;
}
