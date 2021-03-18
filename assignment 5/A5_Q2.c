#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(){
    const int n = 10000000;
    double* a = (double *)malloc(n * sizeof(double));

    double t = clock();
    for (int i = 0; i < n; i++)
    {
        a[i] = (double)i / n;
    }

    t = 1000 * (clock()-t) / CLOCKS_PER_SEC;

    for(int i = 0; i < 5; i++){
        printf("a[%d]: %.7f\n", i, a[i]);
    }
    printf("...\n");
    for(int i = 0; i < 5; i++){
        printf("a[%d]: %.7f\n", n-1-i, a[n-1-i]);
    }
    printf("CPU time: %f\n", t);

    free(a);
    return 0;
}

//CPU time: 50.78ms