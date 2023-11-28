#include <stdio.h>
#include <stdlib.h>
#define MAX_COLOR 255
#define BRIGHTNESS_FACTOR 25

int image_bright(char imageFileName[100], unsigned char *header, unsigned char *colorTable, int size, unsigned char *buffer)
{
	// FILE *fOut = fopen("out/lena_bright.bmp", "w+"); // Output File name
	char ImageFilePath[150];
	sprintf(ImageFilePath, "out/%s/image_bright.bmp", imageFileName);
	FILE *fOut = fopen(ImageFilePath, "w+"); // Output File name

	fwrite(header, sizeof(unsigned char), 54, fOut); // write the header back

	int bitDepth = *(int *)&header[28];

	int i, temp;
	if (bitDepth <= 8) // if ColorTable present, extract it.
	{
		fwrite(colorTable, sizeof(unsigned char), 1024, fOut);
	}
	unsigned char image_buffer[size];

	#pragma omp parallel for schedule(guided, 10)
	for (i = 0; i < size; i++) // increasing pixel values to get image bright
	{
		temp = buffer[i] + BRIGHTNESS_FACTOR;
		image_buffer[i] = (temp > MAX_COLOR) ? MAX_COLOR : temp;
	}

	fwrite(image_buffer, sizeof(unsigned char), size, fOut); // write back to the output image

	fclose(fOut);

	return 0;
}
