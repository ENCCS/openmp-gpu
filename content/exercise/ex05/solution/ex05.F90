! Copyright (c) 2021 ENCCS
program dotproduct
  implicit none

  integer, parameter :: nx = 102400
  real, parameter :: r=0.2

  real, dimension(nx) :: vecA,vecB,vecC
  real    :: sum
  integer :: i

  ! Initialization of vectors
  do i = 1, nx
     vecA(i) = r**(i-1)
     vecB(i) = 1.0
  end do

  ! Dot product of two vectors 
  !$omp target data map(from:vecC) 
  !$omp target map(to:vecA,vecB)
  do i = 1, nx
     vecC(i) =  vecA(i) * vecB(i)
  end do
  !$omp end target 

  ! Initialization of vectors again
  do i = 1, nx
     vecA(i) = 0.5 
     vecB(i) = 2.0
  end do

  !$omp target map(to:vecA,vecB)
  do i = 1, nx
     vecC(i) =  vecC(i) + vecA(i) * vecB(i)
  end do
  !$omp end target
  !$omp end target data 

  sum = 0.0
  ! Calculate the sum
  do i = 1, nx
     sum =  vecC(i) + sum
  end do
  write(*,'(A,F18.6)') 'The sum is: ', sum

end program dotproduct
