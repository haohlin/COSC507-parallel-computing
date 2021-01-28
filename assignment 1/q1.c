#include <stdio.h>
#include <stdlib.h>

int arrLen = 4;

int *GetArray();
void GetAboveAverage(int *arr, float *avg, int *overAvg);

int main(){
    int overAvg = 0;
    float avg;
    int *arr = NULL;

    printf("Enter 4 integers separated by spaces: ");
    // arr = malloc(len * sizeof(int));
    // printf("%d", sizeof(arr));
    arr = GetArray();
    GetAboveAverage(arr, &avg, &overAvg);
    printf("There is %d entry above the average (%.1f)", overAvg, avg);
    free(arr);

    return 0;
}

int *GetArray(){
    int *arr = (int *)malloc(arrLen * sizeof(int));
    if(arr == NULL){
        printf("memory allocation failed.");
    }

    for(int i=0; i<arrLen; i++){
        scanf("%d", &arr[i]);
    }
    return arr;
}

void GetAboveAverage(int *arr, float *avg, int *overAvg){
    int sum = 0;
    for(int i=0; i<arrLen; i++){
        sum += arr[i];
    }

    *avg = (float) sum / arrLen;
    for(int j=0; j<arrLen; j++){
        if (arr[j] > *avg) (*overAvg)++;
    }
}