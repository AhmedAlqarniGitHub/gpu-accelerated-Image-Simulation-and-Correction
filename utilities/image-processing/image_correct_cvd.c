#include <stdio.h>
#include <time.h>
#include "color_blindess.c"

int correct_cvd_protanopia(char imageFileName[100], unsigned char header[54], int size, unsigned char buffer[size][3], int bitDepth, unsigned char colorTable[1024])
{

	char ImageFilePath[150];
	sprintf(ImageFilePath, "out/%s/correct_protanopia.bmp", imageFileName);
	FILE *fOut = fopen(ImageFilePath, "w+"); // Output File name

	fwrite(header, sizeof(unsigned char), 54, fOut); // write the header back
	unsigned char out[size][3];						 // store the output image data
	unsigned int index;

	unsigned char r, g, b;
	float rr, gg, bb;
	float l, m, s;	  // original
	float ll, mm, ss; // updated

	for (index = 0; index < size; index++)
	{
		r = buffer[index][0]; // red
		g = buffer[index][1]; // green
		b = buffer[index][2]; // blue

		// convert to lms
		l = (17.8824 * r) + (43.5161 * g) + (4.11935 * b);	  // long
		m = (3.45565 * r) + (27.1554 * g) + (3.86714 * b);	  // medium
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

#pragma omp parallel for num_threads(1)
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
	unsigned char out[size][3];						 // store the output image data
	unsigned int index;

	unsigned char r, g, b;
	float rr, gg, bb;
	float l, m, s;	  // original
	float ll, mm, ss; // updated

	for (index = 0; index < size; index++)
	{
		r = buffer[index][0]; // red
		g = buffer[index][1]; // green
		b = buffer[index][2]; // blue

		// convert to lms
		l = (17.8824 * r) + (43.5161 * g) + (4.11935 * b);	  // long
		m = (3.45565 * r) + (27.1554 * g) + (3.86714 * b);	  // medium
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

#pragma omp parallel for num_threads(1)
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
	unsigned char out[size][3];						 // store the output image data
	unsigned int index;

	unsigned char r, g, b;
	float rr, gg, bb;
	float l, m, s;	  // original
	float ll, mm, ss; // updated
#pragma omp parallel for schedule(guided, 10) private(r,g,b,rr,gg,bb,l,m,s,ll,mm,ss)
	for (index = 0; index < size; index++)
	{
		r = buffer[index][0];	  // red
		g = buffer[index][1]; // green
		b = buffer[index][2]; // blue

		// convert to lms
		l = (17.8824 * r) + (43.5161 * g) + (4.11935 * b);	  // long
		m = (3.45565 * r) + (27.1554 * g) + (3.86714 * b);	  // medium
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
#pragma omp parallel for num_threads(1)
	for (index = 0; index < size; index++)
	{
		putc(out[index][2], fOut);
		putc(out[index][1], fOut);
		putc(out[index][0], fOut);
	}

	fclose(fOut);
	return 0;
}
