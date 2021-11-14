#include <stdio.h>
#include <math.h>
#define NX 102400

int main(void)
{
  double vecA[NX],vecB[NX],vecC[NX];
  double r=0.2;

/* Initialization of vectors */
  for (int i = 0; i < NX; i++) {
     vecA[i] = pow(r, i);
     vecB[i] = 1.0;
  }

/* Dot product of two vectors */
  for (int i = 0; i < NX; i++) {
     vecC[i] = vecA[i] * vecB[i];
  }

  double sum = 0.0;
  /* Calculate the sum */
  for (int i = 0; i < NX; i++) {
    sum += vecC[i];
  }
  printf("The sum is: %8.6f \n", sum);
  return 0;
}
