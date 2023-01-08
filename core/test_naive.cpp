#include "matrix.h"
#include <ctime>
#include <stdio.h>


int main()
{
    printf("HI\n");
    clock_t start = 0, end = 0;
    Matrix a(1024, 1024), b(1024,1024);

    start = clock();
    multiply_naive(a, b);
    end = clock();
    printf("multiply_naive : %f\n", double(end-start) / CLOCKS_PER_SEC);

    start = clock();
    multiply_naive_reorder(a, b);
    end = clock();
    printf("multiply_naive_reorder : %f\n", double(end-start) / CLOCKS_PER_SEC);

    start = clock();
    multiply_naive_pthread(a, b, 4);
    end = clock();
    printf("multiply_naive_pthread : %f\n", double(end-start) / CLOCKS_PER_SEC);
    

    start = clock();
    multiply_naive_omp(a, b, 4);
    end = clock();
    printf("multiply_naive_omp : %f\n", double(end-start) / CLOCKS_PER_SEC);
}