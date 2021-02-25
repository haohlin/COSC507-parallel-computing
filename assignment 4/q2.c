#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NRA 20  /* number of rows in A */ 
#define NCA 30  /* number of columns in A = number of rows in B */ 
#define NCB 10  /* number of columns in matrix B */ 

int main(){

    /* Creating matrices */
    int **a = (int**)malloc(sizeof(int*) * NRA);
    int **b = (int**)malloc(sizeof(int*) * NCA);
    int **c = (int**)malloc(sizeof(int*) * NRA);
    for (int i = 0; i < NRA; i++)
    {
        a[i] = (int*)malloc(sizeof(int) * NCA);
        c[i] = (int*)malloc(sizeof(int) * NCB);
    }
    for (int i = 0; i < NCA; i++)
    {
        b[i] = (int*)malloc(sizeof(int) * NCB);
    }

    
    # pragma omp parallel num_threads(12)
    {
        /* Initializing matrices */
        # pragma omp for nowait
        for (int i = 0; i < NRA; i++)
        {
            for (int j = 0; j < NCA; j++)
            {
                a[i][j] = i + j;
            }
        }

        # pragma omp for
        for (int i = 0; i < NCA; i++)
        {
            for (int j = 0; j < NCB; j++)
            {
                b[i][j] = i * j + 1;
            }
        }

        /* Matrix multiplication */
        # pragma omp for
        for (int i = 0; i < NRA; i++)
        {
            for (int j = 0; j < NCB; j++)
            {
                for (int k = 0; k < NCA; k++)
                {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
    

    printf("[");
    for (int i = 0; i < NRA; i++)
    {
        printf("[");
        for (int j = 0; j < NCB; j++)
        {
            printf("%d\t", c[i][j]);
        }
        
        printf("],\n");
    }
    printf("]\n");
    
    return 0;
}