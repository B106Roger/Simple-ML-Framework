#include "matrix.h"
#include <ctime>
#include <stdio.h>


int main()
{
    printf("HI\n");
    clock_t start = 0, end = 0;
    Matrix a(1024, 1024), b(1024,1024);

    // start = clock();
    // multiply_naive(a, b);
    // end = clock();
    // printf("multiply_naive : %f\n", double(end-start) / CLOCKS_PER_SEC);

    // start = clock();
    // multiply_tile_modify(a, b, 16);
    // end = clock();
    // printf("multiply_tile_modify : %f\n", double(end-start) / CLOCKS_PER_SEC);


    start = clock();
    multiply_tile_modify_pthread(a, b, 16, 8, 8);
    end = clock();
    printf("multiply_tile_modify_pthread : %f\n", double(end-start) / CLOCKS_PER_SEC);
}