module example
  real(8), allocatable :: A(:)

  contains
    subroutine initialize(N)
      integer :: N

      allocate(A(N))
      !$omp target enter data map(alloc:A)

    end subroutine initialize

    subroutine finalize()

      !$omp target exit data map(delete:A)
      deallocate(A)

    end subroutine finalize
end module example
