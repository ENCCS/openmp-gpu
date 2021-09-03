Data environment
================

.. objectives::

   - Understand explicit and implicit data movement
   - Understand structured and unstructured data clauses
   - Understand different mapping types



Data mapping
------------

Due to distinct memory spaces on host and device, transferring data
becomes inevitable. A combination of both explicit and implicit data
mapping is used. 

The ``MAP`` cluase on a device construct explicitly
specifies how items are mapped from the host to the device data
environment.  The common mapped items consist of arrays(array
sections), scalars, pointers, and structure elements.  The various
forms of the map cluase are summarised in the following table


.. csv-table::
   :widths: auto
   :delim: ;

   ``map([map-type]:list)`` ; :doc:`map clause`
   ``map(to:list)`` ; :doc:`On entering the region, variables in the list are initialized on the device using the original values from the host`
   ``map(from:list)`` ;  :doc:`At the end of the target region, the values from variables in the list are copied into the original variables on the host. On entering the region, the initial value of the variables on the device is not initialized`       
   ``map(tofrom:list)`` ; :doc:`the effect of both a map-to and a map-from`
   ``map(alloc:list)`` ;  :doc:`On entering the region, data is allocated and uninitialized on the device`
   ``map(list)`` ; :doc:`equivalent to ``map(tofrom:list)```

.. +---------------------------+-----------------------------------------------+
   |                           |                                               |
   +===========================+===============================================+
   |  ``map([map-type]:list)`` | map clause                                    |
   +---------------------------+-----------------------------------------------+
   |  ``map(to:list)``         | On entering the region, variables in the list |
   |                           | are initialized on the device using the       |
   |                           | original values from the host                 |
   +---------------------------+-----------------------------------------------+
   |  ``map(from:list)``       | At the end of the target region, the values   |
   |                           | from variables in the list are copied into    |
   |                           | the original variables on the host. On        |
   |                           | entering the region, the initial value of the |
   |                           | variables on the device is not initialized    |
   +---------------------------+-----------------------------------------------+
   |  ``map(tofrom:list)``     | the effect of both a map-to and a map-from    |
   +---------------------------+-----------------------------------------------+
   |  ``map(alloc:list)``      | On entering the region, data is allocated and |
   |                           | uninitialized on the device                   |
   +---------------------------+-----------------------------------------------+
   |  ``map(list)``            | equivalent to ``map(tofrom:list)``            |
   +---------------------------+-----------------------------------------------+
   

	
If the variables are not explicitly mapped, the compiler will do it implicitly:
  - Since v4.5, scalar is mapped as firstprivate, and the variable is not copied back to the host
  - non-scalar variables are mapped with a map-type tofrom
  - a C/C++ pointer is mapped as a zero-length array section
  - note that only the pointer value is mapped, but not the data it points to


.. note::

	When mapping data arrays or pointers, be careful about the array section notation:
	  - In C/C++: array[lower-bound:length]. The notation :N is equivalent to 0:N.
	  - In Fortran:array[lower-bound:upper-bound]. The notation :N is equivalent to 1:N.



.. challenge:: Example: implicit data mapping 

   .. tabs::

      .. tab:: C/C++

             .. code-block:: c
             	:linenos:
             	:emphasize-lines: 8

			extern void init(float*, float*, int);
			extern void output(float*, int);
			void vec_mult(int N)
			{
			   int i;
			   float p[N], v1[N], v2[N];
			   init(v1, v2, N);
			   #pragma omp target
			   #pragma omp parallel for private(i)
			   for (i=0; i<N; i++)
			     p[i] = v1[i] * v2[i];
			   output(p, N);
			}


      .. tab:: Fortran

             .. code-block:: fortran
             	:linenos:
             	:emphasize-lines: 8,13

			subroutine vec_mult(N)
			   integer ::  i,N
			   real    ::  p(N), v1(N), v2(N)


			   call init(v1, v2, N)

			   !$omp target
			   !$omp parallel do
			   do i=1,N
			      p(i) = v1(i) * v2(i)	
			   end do
			   !$omp end target

			   call output(p, N)
			end subroutine

Data region
-----------

How the ``TARGET`` construct creates storage, transfer data, and remove
storage on the device are clasiffied as two categories: structured
data region and unstructured data region.

Structured Data Regions
-----------------------

The ``TARGET DATA`` construct is used to create a structured data region
which is convenient for providing persistent data on the device which
could be used for subseqent target constructs.

.. challenge:: Syntax

   .. tabs::

      .. tab:: C/C++

             .. code-block:: c

		  #pragma omp target data clause [clauses]
		  	structured-block

             .. code-block:: c

	          clause:
                  if( [target data:]scalar-logical-expression)
                  device(scalar-integer-expression)
		  map([map-type :] list)
		  use_device_ptr(list)


      .. tab:: Fortran

             .. code-block:: fortran

                  !$omp target data clause [clauses]
			  structured-block
		  !$omp end target data

             .. code-block:: fortran

	          clause:
                  if( [target data:]scalar-logical-expression)
                  device(scalar-integer-expression)
		  map([map-type :] list)
		  use_device_ptr(list)





.. challenge:: Example: ``TARGET DATA`` structured region 

   .. tabs::

      .. tab:: C/C++

             .. code-block:: c
             	:linenos:
             	:emphasize-lines: 8,10,15

			extern void init(float*, float*, int);
			extern void init_again(float*, float*, int);
			extern void output(float*, int);
			void vec_mult(float *p, float *v1, float *v2, int N)
			{
			   int i;
			   init(v1, v2, N);
			   #pragma omp target data map(from: p[0:N])
			   {
			      #pragma omp target map(to: v1[:N], v2[:N])
			      #pragma omp parallel for
			      for (i=0; i<N; i++)
			        p[i] = v1[i] * v2[i];
			      init_again(v1, v2, N);
			      #pragma omp target map(to: v1[:N], v2[:N])
			      #pragma omp parallel for
			      for (i=0; i<N; i++)
			        p[i] = p[i] + (v1[i] * v2[i]);
			   }
			   output(p, N);
			}



      .. tab:: Fortran

             .. code-block:: fortran
             	:linenos:
             	:emphasize-lines: 5,6,12

			subroutine vec_mult(p, v1, v2, N)
			   real    ::  p(N), v1(N), v2(N)
			   integer ::  i
			   call init(v1, v2, N)
			   !$omp target data map(from: p)
			      !$omp target map(to: v1, v2 )
			         !$omp parallel do
			         do i=1,N
			            p(i) = v1(i) * v2(i)
			         end do
			      !$omp end target
			      call init_again(v1, v2, N)
			      !$omp target map(to: v1, v2 )
			         !$omp parallel do
			         do i=1,N
			            p(i) = p(i) + v1(i) * v2(i)
			         end do
			      !$omp end target
			   !$omp end target data
			   call output(p, N)
			end subroutine	




Unstructured Data Regions
-------------------------

The ``TARGET DATA`` construct however is inconvenient in real applications.
The unstructured data constructs (``TARGET ENTER DATA`` and ``TARGET EXIT DATA``) 
have much more freedom in creating and deleting of data on the device at any appropriate point.


.. challenge:: Syntax

   .. tabs::

      .. tab:: C/C++

             .. code-block:: c

		  #pragma omp target enter data [clauses]

	     .. code-block:: c

		  #pragma omp target exit data [clauses]

             .. code-block:: c

	          clause:
                  if(scalar-logical-expression)
                  device(scalar-integer-expression)
		  map([map-type :] list)
		  depend(dependence-type:list)
		  nowait


      .. tab:: Fortran

             .. code-block:: fortran

                  !$omp target enter data [clauses]

             .. code-block:: fortran

		  !$omp target exit data [clauses]

             .. code-block:: fortran

	          clause:
                  if(scalar-logical-expression)
                  device(scalar-integer-expression)
		  map([map-type :] list)
		  depend(dependence-type:list)
		  nowait



.. keypoints::

  Structured Data Region
    - start and end points within a single subroutine
    - Memory exist within the data region

  Unstructured Data Region
    - multiple start and end points across different subroutines
    - Memory exists until explicitly deallocated




DECLARE TARGET construct
------------------------
The ``DECLARE TARGET`` construct is used to create a device executable version of the subroutine/function. 
Another typical usage of   ``DECLARE TARGET`` construct is to define global variables to be accessed on the devices.


.. challenge:: Syntax

   .. tabs::

      .. tab:: C/C++

             .. code-block:: c

		  #pragma omp declare target 
		  	declarations-definition-seq
		  #pragma omp end declare target

		  or

                  #pragma omp declare target (extended-list)

                  or

		  #pragma omp declare target clause [clauses]

             .. code-block:: c

	          clause:
                  to(extended-list)
                  link(list)


      .. tab:: Fortran

             .. code-block:: fortran

                  !$omp declare target (extended-list)

		  or

		  !$omp declare target [clauses]

             .. code-block:: fortran

	          clause:
                  to(extended-list)
                  link(list)

.. note::

	extended-list: A comma-separated list of named variables, procedure names, and named common blocks.




.. challenge:: Example: ``DECLARE TARGET`` 

   .. tabs::

      .. tab:: C/C++

             .. code-block:: c
             	:linenos:
             	:emphasize-lines: 2-6

			#define N 10000
			#pragma omp declare target
			float Q[N][N];
			float Pfun(const int i, const int k)
			{ return Q[i][k] * Q[k][i]; }
			#pragma omp end declare target
			float accum(int k)
			{
			    float tmp = 0.0;
			    #pragma omp target update to(Q)
			    #pragma omp target map(tofrom: tmp)
			    #pragma omp parallel for reduction(+:tmp)
			    for(int i=0; i < N; i++)
			        tmp += Pfun(i,k);
			    return tmp;
			}


      .. tab:: Fortran

             .. code-block:: fortran
             	:linenos:
             	:emphasize-lines: 2-11

			module my_global_array
			!$omp declare target (N,Q)
			integer, parameter :: N=10
			real               :: Q(N,N)
			contains
			function Pfun(i,k)
			!$omp declare target
			real               :: Pfun
			integer,intent(in) :: i,k
			   Pfun=(Q(i,k) * Q(k,i))
			end function
			end module

			function accum(k) result(tmp)
			use my_global_array
			real    :: tmp
			integer :: i, k
			   tmp = 0.0e0
			   !$omp target map(tofrom: tmp)
			   !$omp parallel do reduction(+:tmp)
			   do i=1,N
			      tmp = tmp + Pfun(k,i)
			   end do
			   !$omp end target
			end function




TARGET UPDATE construct
------------------------
The ``TARGET UPDATE`` construct is used to keep the variable consistent between the host and the device.
Data can be updated within a target regions  with the transfer direction specified in the clause.

.. challenge:: Syntax

   .. tabs::

      .. tab:: C/C++

             .. code-block:: c

		  #pragma omp target update [clause] 


             .. code-block:: c

	          clause is motion-clause or one of:
                  if(scalar-logical-expression)
                  device(scalar-integer-expression)
		  nowait
                  depend(dependence-type:list)

                  motion-clause:
		  to(list)
		  from(list)


      .. tab:: Fortran

             .. code-block:: fortran

                  !$omp target udpate clause


             .. code-block:: fortran

	          clause is motion-clause or one of:
                  if(scalar-logical-expression)
                  device(scalar-integer-expression)
		  nowait
                  depend(dependence-type:list)

                  motion-clause:
		  to(list)
		  from(list)



Optimize Data Transfers
-----------------------

- Explicitely map the data instead of using the implicit mapping
- Reduce the amount of data mapping between host and device, get 
  rid of unneeded data transfer
- Try to keep data environment residing on the target device as long
  as possible




.. exercise:: Exercise: Data Movement

   This exercise is about optimization and explicitly moving the data using  
   the "target data" family constructs. Three incomplete functions are added  
   to explicitly move the data around in core.cpp or core.F90. You need to
   add the directives for data movement for them.

   The exercise is under /content/exercise/data_mapping

.. solution::

   .. tabs::

      .. tab:: C++

         .. literalinclude:: exercise/solution/data_mapping/core.cpp
                        :language: cpp
			:emphasize-lines: 59-60,74-75,88


      .. tab:: Fortran

         .. literalinclude:: exercise/solution/data_mapping/fortran/core.F90
                        :language: fortran
                        :emphasize-lines: 58,72,84



