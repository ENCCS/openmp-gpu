program hello

#ifdef _OPENMP
  use omp_lib
#endif
  implicit none

  integer :: num_devices,nteams,nthreads
  logical :: initial_device

  num_devices = omp_get_num_devices()
  print *, "Number of available devices", num_devices

  !$omp target map(nteams,nthreads)
  !$omp teams num_teams(10) thread_limit(4)
  !$omp parallel
    initial_device = omp_is_initial_device()
    nteams= omp_get_num_teams()
    nthreads= omp_get_num_threads()
  !$omp end parallel 
  !$omp end teams
  !$omp end target 
    if (initial_device) then
      write(*,*) "Running on host"
    else 
      write(*,'(A,I4,A,I4,A)') "Running on device with ",nteams, " teams in total and ", nthreads, " threads in each team"
    end if

end program

