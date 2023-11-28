#include <stdio.h>
#include <time.h>
#define MAX_VALUE 255

int image_colortosepia(char imageFileName[100], unsigned char header[54], int size, unsigned char buffer[size][3], int bitDepth, unsigned char colorTable[1024])
{
	char ImageFilePath[150];
	sprintf(ImageFilePath, "out/%s/sepia.bmp", imageFileName);
	FILE *fOut = fopen(ImageFilePath, "w+"); // Output File name

	int i, r, g, b;

	fwrite(header, sizeof(unsigned char), 54, fOut); // write the header back

#pragma omp parallel for num_threads(1)
	for (i = 0; i < size; i++)
	{
		r = 0;
		g = 0;
		b = 0;

		// conversion formula of rgb to sepiabuffer
		r = (buffer[i][2] * 0.393) + (buffer[i][1] * 0.769) + (buffer[i][0] * 0.189);
		g = (buffer[i][2] * 0.349) + (buffer[i][1] * 0.686) + (buffer[i][0] * 0.168);
		b = (buffer[i][2] * 0.272) + (buffer[i][1] * 0.534) + (buffer[i][0] * 0.131);

		if (r > MAX_VALUE)
		{ // if value exceeds
			r = MAX_VALUE;
		}
		if (g > MAX_VALUE)
		{
			g = MAX_VALUE;
		}
		if (b > MAX_VALUE)
		{
			b = MAX_VALUE;
		}
		putc(b, fOut);
		putc(g, fOut);
		putc(r, fOut);
	}
	fclose(fOut);

	return 0;
}
