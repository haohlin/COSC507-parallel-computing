/**************************************************************

 The program reads a BMP image file and creates a new
 image that is the negative or desaturated of the input file.

 **************************************************************/

// To compile: gcc -c qdbmp.c && gcc -c -fopenmp main.c && gcc -fopenmp  qdbmp.o main.o -o main && ./main

#include "qdbmp.h"
#include <stdio.h>
#include <omp.h>

typedef enum {desaturate, negative} ImgProcessing ;

/* Creates a negative image of the input bitmap file */
int main() {
	const char* inFile = "okanagan.bmp";
	const char* outFile = "okanagan_processed.bmp";
	const ImgProcessing processingType = desaturate; //or negative

	UCHAR r, g, b;
	UINT width, height;
	UINT x, y;
	BMP* bmp;

	/* Read an image file */
	bmp = BMP_ReadFile(inFile);
	BMP_CHECK_ERROR(stdout, -1);

	/* Get image's dimensions */
	width = BMP_GetWidth(bmp);
	height = BMP_GetHeight(bmp);

	/* Input number of threads */
	int n_threads;
	printf("Input number of threads:\n");
	scanf("%d", &n_threads);
	if (height%n_threads != 0)
	{
		fprintf(stderr, "warning: Recommend number of threads divisible by image height.\n");
	}
	
	int seg_lenth = height / n_threads;
	
	double t = omp_get_wtime();
	omp_set_nested(1);
	/* Iterate through all the image's pixels */
	#pragma omp parallel for num_threads(n_threads) private(r,g,b,x,y)
	for (x = 0; x < width; ++x) {
		#pragma omp parallel for num_threads(n_threads) private(r,g,b,x,y)
		for (y = 0; y < height; ++y) {
			/* Get pixel's RGB values */
			BMP_GetPixelRGB(bmp, x, y, &r, &g, &b);

			/* Write new RGB values */
			if(processingType == negative)
				BMP_SetPixelRGB(bmp, x, y, 255 - r, 255 - g, 255 - b);
			else if(processingType == desaturate){
				UCHAR gray = r * 0.3 + g * 0.59 + b * 0.11;
				BMP_SetPixelRGB(bmp, x, y, gray, gray, gray);
			}
		}
	}
	/* calculate and print processing time*/
	t = 1000 * (omp_get_wtime() - t);
	printf("Finished image processing in %.1f ms.", t);

	/* Save result */
	BMP_WriteFile(bmp, outFile);
	BMP_CHECK_ERROR(stdout, -2);

	/* Free all memory allocated for the image */
	BMP_Free(bmp);

	return 0;
}

/*  
num_threads	no_nest_exe_time	nest_exe_time
2			265.7ms				267.4ms
4			242.2ms				993.3ms
8			211.7ms				451.4ms
16			153.1ms				912.5ms
24			151.5ms				1444.5ms

Nested solution is not better as it adds overheads.
*/