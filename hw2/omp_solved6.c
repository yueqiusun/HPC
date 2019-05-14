/* we don't want the to fork twice both outside and inside the dotprod function. 
Get rid of the parallel statement outside the dotprod function.

/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];

float dotprod ()
{
int i,tid;
float sum1 = 0;


#pragma omp parallel shared(sum1) private(tid, i) 
{
	tid = omp_get_thread_num();
	#pragma omp for reduction(+:sum1)
    for (i=0; i < VECLEN; i++)
    {
    sum1 = sum1 + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",tid,i);
    }
}
return sum1;
}



int main (int argc, char *argv[]) {
int i;
float sum;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

//#pragma omp parallel shared(sum)
  sum = dotprod();

printf("Sum = %f\n",sum);

}

