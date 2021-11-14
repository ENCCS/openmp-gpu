program dotproduct
  implicit none

  integer :: x

  x = 0
  !$omp target data map(tofrom:x) 
  ! check point 1 
  x = 10                        
  ! check point 2 
  !$omp target update to(x)       
  ! check point 3 
  !$omp end target data

end program dotproduct
