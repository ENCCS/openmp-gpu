! Copyright (c) 2019 CSC Training
! Copyright (c) 2021 ENCCS
! PNG writer for heat equation solver
module pngwriter
  use heat

contains

  function save_png(data, nx, ny, fname) result(stat)

    use, intrinsic :: ISO_C_BINDING
    implicit none

    real(dp), dimension(:,:), intent(in) :: data
    integer, intent(in) :: nx, ny
    character(len=*), intent(in) :: fname
    integer :: stat

    ! Interface for save_png C-function
    interface
       ! The C-function definition is
       !   int save_png(double *data, const int nx, const int ny,
       !                const char *fname)
       function save_png_c(data, nx, ny, fname, order) &
            & bind(C,name="save_png") result(stat)
         use, intrinsic :: ISO_C_BINDING
         implicit none
         real(kind=C_DOUBLE) :: data(*)
         integer(kind=C_INT), value, intent(IN) :: nx, ny
         character(kind=C_CHAR), intent(IN) :: fname(*)
         character(kind=C_CHAR), value, intent(IN) :: order
         integer(kind=C_INT) :: stat
       end function save_png_c
    end interface

    stat = save_png_c(data, nx, ny, trim(fname) // C_NULL_CHAR, 'f')
    if (stat /= 0) then
       write(*,*) 'save_png returned error!'
    end if

  end function save_png

end module pngwriter
