#include <stdio.h>
int main(void)
{
  int x = 0;

  #pragma omp target data map(tofrom:x)
  {
/* check point 1 */
    x = 10;                        
/* check point 2 */
  #pragma omp target update to(x)       
/* check point 3 */
  }

return 0;
}
