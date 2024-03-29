! Copyright (c) 2019 CSC Training
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
  !$omp target teams distribute map(from:vecC) map(to:vecA,vecB) 
  do i = 1, nx
     vecC(i) =  vecA(i) * vecB(i)
  end do
  !$omp end target teams distribute

  sum = 0.0
  ! Calculate the sum
  !$omp target map(tofrom:sum)
  do i = 1, nx
     sum =  vecC(i) + sum
  end do
  !$omp end target
  write(*,*) 'The sum is: ', sum

end program dotproduct
