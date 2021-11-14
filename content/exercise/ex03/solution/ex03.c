#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int main() 
{
  int num_devices = omp_get_num_devices();
  printf("Number of available devices %d\n", num_devices);

  #pragma omp target 
  #pragma omp teams num_teams(2) thread_limit(3)
  #pragma omp parallel
  {
      if (omp_is_initial_device()) {
        printf("Running on host\n");    
      } else {
        int nteams= omp_get_num_teams(); 
        int nthreads= omp_get_num_threads();
        printf("Running on device with %d teams in total and %d threads in each team\n",nteams,nthreads);
      }
  }
  
}

