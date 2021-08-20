! Main solver routines for heat equation solver
module core
  use heat

contains

  ! Update the temperature values using five-point stencil
  ! Arguments:
  !   curr (type(field)): current temperature values
  !   prev (type(field)): temperature values from previous time step
  !   a (real(dp)): diffusivity
  !   dt (real(dp)): time step
  subroutine evolve(curr, prev, a, dt)

    implicit none

    type(field), intent(inout) :: curr, prev
    real(dp) :: a, dt
    integer :: i, j, nx, ny

    ! Help the compiler avoid being confused
    nx = curr%nx
    ny = curr%ny

    ! Determine the temperature field at next time step As we have
    ! fixed boundary conditions, the outermost gridpoints are not
    ! updated.
    do j = 1, ny
       do i = 1, nx
          curr%data(i, j) = prev%data(i, j) + a * dt * &
               & ((prev%data(i-1, j) - 2.0 * prev%data(i, j) + &
               &   prev%data(i+1, j)) / curr%dx**2 + &
               &  (prev%data(i, j-1) - 2.0 * prev%data(i, j) + &
               &   prev%data(i, j+1)) / curr%dy**2)
       end do
    end do
  end subroutine evolve

end module core
