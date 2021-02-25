#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <omp.h>

void count_sort(int a[], int n);

int main(){
    clock_t start, end;
    int n = 10000;
    int *a = (int*)malloc(sizeof(int) * n);

	double t = omp_get_wtime();
    count_sort(a, n);

	t = 1000 * (omp_get_wtime() - t);
	printf("Execution time: %.1f ms.", t);

    return 0;
}

void count_sort(int a[], int n) { 
    int i, j, count; 
    int* temp = malloc(n * sizeof(int)); 

    # pragma omp parallel for private(j, count) num_threads(8)
    for (i = 0; i < n; i++){ 
        //count all elements < a[i] 
        count = 0; 
        for (j = 0; j < n; j++) 
            if(a[j]<a[i] ||(a[j]==a[i] && j<i))	 
                count++; 
        //place a[i] at right order 
        temp[count] = a[i]; 
    } 
    memcpy(a, temp, n * sizeof(int)); 
    free(temp); 
} 

/*
1. j and count should private, temp remain global
2. No loop-carried dependences. For each specific a[i], there is only
   one possible count, therefore different order of a[i] makes no difference.
3. serial: 274.20ms
   parallel (8 threads): 34.2ms
*/