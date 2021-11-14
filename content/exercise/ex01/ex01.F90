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
  do i = 1, nx
     vecC(i) =  vecA(i) * vecB(i)
  end do

  sum = 0.0
  ! Calculate the sum 
  do i = 1, nx
     sum =  vecC(i) + sum
  end do

  write(*,*) 'The sum is: ', sum

end program dotproduct
