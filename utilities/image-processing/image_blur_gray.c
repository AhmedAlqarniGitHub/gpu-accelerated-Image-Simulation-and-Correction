
#include <stdio.h>
#include <string.h>

int image_bluring_gray(char imageFileName[100], unsigned char header[54], int size, int height, int width, unsigned char buffer[size], int bitDepth, unsigned char colorTable[1024])
{
	char ImageFilePath[150];
	sprintf(ImageFilePath, "out/%s/image_bluring_gray.bmp", imageFileName);
	FILE *fOut = fopen(ImageFilePath, "w+"); // Output File name

	// FILE *fOut = fopen("out/bluring_gray.bmp", "w+"); // Output File name

	int i, j, y, x;

	fwrite(header, sizeof(unsigned char), 54, fOut); // write the header back

	if (bitDepth <= 8) // if ColorTable present, extract it.
	{
		fwrite(colorTable, sizeof(unsigned char), 1024, fOut);
	}

	unsigned char out[size]; // to store the image data
	memcpy(out, buffer, size);

	float v = 1.0 / 9.0;
	float kernel[3][3] = {{v, v, v}, // initialize the blurrring kernel
						  {v, v, v},
						  {v, v, v}};

	for (x = 1; x < height - 1; x++)
	{
		for (y = 1; y < width - 1; y++)
		{
			float sum = 0.0;
			for (i = -1; i <= 1; ++i)
			{
				for (j = -1; j <= 1; ++j)
				{
					sum = sum + (float)kernel[i + 1][j + 1] * buffer[(x + i) * width + (y + j)]; // matrix multiplication with kernel
				}
			}
			out[(x * width) + y] = sum;
		}
	}

	fwrite(out, sizeof(unsigned char), size, fOut); // write image data back to the file
	fclose(fOut);
	return 0;
}
