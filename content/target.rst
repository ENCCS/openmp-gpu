Offloading to GPU
=================

.. objectives::

   - Understand and be able to offload code to device
   - Understand different constructs to create parallelism on device



.. _host_device_model:

Host-device model
-----------------

Since version 4.0 , OpenMP supports heterogeneous systems. OpenMP uses
``TARGET`` construct to offload execution from the host to the target
device(s), and hence the directive name. In addition, the associated
data needs to be transferred to the device(s) as well.  Once
transferred, the target device owns the data and accesses by the host
*during the execution* of the target region is forbidden.

A host/device model is generally used by OpenMP for offloading:

  - normally there is only one single host: e.g. CPU
  - one or multiple target devices *of the same kind*: e.g. coprocessor, GPU, FPGA, ...
  - unless with unified shared memory, the host and device have separate memory address space


.. note::

   Under the following conditions, there will be **NO data transfer** to the device

   - device is host
   - data already exists on the device from a previous execution
     







.. _device_execution_model:

Device execution model
----------------------

The execution on the device is host-centric

1.the host creates the data environments on the device(s)   

2.the host maps data to the device data environment, which is data movement to the device  

3.the host offloads OpenMP target regions to the target device to be  executed  

4.the host transfers data from the device to the host   

5.the host destroys the data environment on the device



TARGET construct
----------------

The ``TARGET`` construct consists of a *target* directive and an
execution region. It is used to transfer both the control flow from
the host to the device and the data between the host and device.

.. challenge:: Syntax

   .. tabs::

      .. tab:: C/C++

             .. code-block:: c

		  #pragma omp target [clauses]
 		       structured-block

             .. code-block:: c

	          clause:
			if([ target:] scalar-expression)
			device(integer-expression) 
			private(list)
			firstprivate(list)	
			map([map-type:] list)
			is_device_ptr(list)
			defaultmap(tofrom:scalar) 
			nowait
			depend(dependence-type : list)


      .. tab:: Fortran

             .. code-block:: fortran

		  !$omp target [clauses]
		        structured-block
		  !$omp end target


             .. code-block:: fortran

	          clause:
			if([ target:] scalar-expression)
			device(integer-expression) 
			private(list)
			firstprivate(list)	
			map([map-type:] list)
			is_device_ptr(list)
			defaultmap(tofrom:scalar) 
			nowait
			depend(dependence-type : list)


.. exercise:: Exercise00: Hello world with OpenMP offloading


   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: exercise/ex00/ex00.c
            :language: c
            :linenos:


      .. tab:: Fortran

	 .. literalinclude:: exercise/ex00/ex00.F90                  
	    :language: fortran
            :linenos:
            

.. exercise:: Exercise01: Adding ``TARGET`` construct


   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: exercise/ex01/ex01.c
            :language: c
            :linenos:


      .. tab:: Fortran

	 .. literalinclude:: exercise/ex01/ex01.F90                  
	    :language: fortran
            :linenos:
            



.. solution:: 

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: exercise/ex01/solution/ex01.c
            :language: c
            :linenos:
	    :emphasize-lines: 17


      .. tab:: Fortran

	 .. literalinclude:: exercise/ex01/solution/ex01.F90                  
	    :language: fortran
            :linenos:
            :emphasize-lines: 18,22


	  




Creating parallelism on the target device
-----------------------------------------

The ``TARGET`` construct transfers the control flow to the device is
sequential and synchronous, and it is because OpenMP separates offload
and parallelism.  One needs to explicitly create parallel regions on
the target device to make efficient use of the device(s).

TEAMS construct
~~~~~~~~~~~~~~~

.. challenge:: Syntax

   .. tabs::

      .. tab:: C/C++

             .. code-block:: c

		  #pragma omp teams [clauses]
		  	structured-block
		  
             .. code-block:: c

	          clause:
                  num_teams(integer-expression)
                  thread_limit(integer-expression)
		  default(shared | none)
		  private(list)
      		  firstprivate(list)
		  shared(list)
		  reduction(reduction-identifier : list)


      .. tab:: Fortran

             .. code-block:: fortran

		  !$omp teams [clauses] 
		          structured-block
		  !$omp end teams

             .. code-block:: fortran

	          clause:
                  num_teams(integer-expression)
                  thread_limit(integer-expression)
		  default(shared | none)
		  private(list)
      		  firstprivate(list)
		  shared(list)
		  reduction(reduction-identifier : list)



The ``TEAMS`` construct creates a league of one-thread teams where 
the thread of each team executes *concurrently* and is in its own *contention group*. 
The number of teams created is implementation defined, but is no more than 
num_teams if specified in the clause. The maximum number of threads participating in 
the contention group that each team initiates is implementation defined as well, 
unless thread_limit is specified in the clause. 
Threads in a team can synchronize but no synchronization among teams. 
The ``TEAMS`` construct must be contained in a ``TARGET`` construct, 
without any other directives, statements or declarations in between.  


.. note:: 

   A contention group is the set of all threads that are descendants of an initial thread.  
   An initial thread is never a descendant of another initial thread. 


DISTRIBUTE construct
~~~~~~~~~~~~~~~~~~~~

.. challenge:: Syntax

   .. tabs::

      .. tab:: C/C++

             .. code-block:: c

		  #pragma omp distribute [clauses]
		  	for-loops
		  
             .. code-block:: c

	          clause:
		  private(list)
      		  firstprivate(list)
		  lastprivate(list)
		  collapse(n)
		  dist_schedule(kind[, chunk_size])


      .. tab:: Fortran

             .. code-block:: fortran

		  !$omp distribute [clauses] 
		          do-loops
		  [!$omp end distribute]

             .. code-block:: fortran

	          clause:
		  private(list)
      		  firstprivate(list)
		  lastprivate(list)
		  collapse(n)
		  dist_schedule(kind[, chunk_size])



The ``DISTRIBUTE`` construct is a coarsely worksharing construct 
which distributes the loop iterations across the master threads in the teams,
but no worksharing within the threads in one team. No implicit barrier
at the end of the construct and no guarantee about the order the teams
will execute.


To further create threads within each team and distritute loop iterations across threads,
we will use the  ``PARALLEL FOR/DO`` constructs.

PARALLEL construct
~~~~~~~~~~~~~~~~~~

.. challenge:: Syntax

   .. tabs::

      .. tab:: C/C++

             .. code-block:: c

		  #pragma omp parallel [clauses]
		  	structured-block
		  
             .. code-block:: c

	          clause:
                  num_threads(integer-expression)
		  default(shared | none)
		  private(list)
      		  firstprivate(list)
		  shared(list)
		  reduction(reduction-identifier : list)


      .. tab:: Fortran

             .. code-block:: fortran

		  !$omp parallel [clauses] 
		          structured-block
		  !$omp end parallel

             .. code-block:: fortran

	          clause:
                  num_threads(integer-expression)
		  default(private | firstprivate | shared | none)
		  private(list)
      		  firstprivate(list)
		  shared(list)
                  copyin(list)
		  reduction(reduction-identifier : list)



FOR/DO construct
~~~~~~~~~~~~~~~~

.. challenge:: Syntax

   .. tabs::

      .. tab:: C/C++

             .. code-block:: c

		  #pragma omp for [clauses]
		  	structured-block
		  
             .. code-block:: c

	          clause:
		  private(list)
      		  firstprivate(list)
      		  lastprivate(list)
		  reduction(reduction-identifier : list)
		  schedule(kind[, chunk_size])
                  collapse(n)

      .. tab:: Fortran

             .. code-block:: fortran

		  !$omp do [clauses] 
		          structured-block
		  [!$omp end do]

             .. code-block:: fortran

	          clause:
		  private(list)
      		  firstprivate(list)
      		  lastprivate(list)
		  reduction(reduction-identifier : list)
		  schedule(kind[, chunk_size])
                  collapse(n)



.. keypoints::

  TEAMS DISTRIBUTE construct
    - Coarser-grained parallelism
    - Spawns multiple teams, each with one thread
    - Threads in different teams can’t synchronize with each othe

  PARALLEL FOR/DO construct
    - Finer-grained parallelism
    - Spawns many threads in a team
    - Threads in a team can synchronize with each other




.. exercise:: Exercise02: Adding constructs for parallelism


   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: exercise/ex02/ex02.c
            :language: c
            :linenos:


      .. tab:: Fortran

	 .. literalinclude:: exercise/ex02/ex02.F90                  
	    :language: fortran
            :linenos:
            

.. solution:: 

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: exercise/ex02/solution/ex02.c
            :language: c
            :linenos:
	    :emphasize-lines: 17


      .. tab:: Fortran

	 .. literalinclude:: exercise/ex02/solution/ex02.F90                  
	    :language: fortran
            :linenos:
            :emphasize-lines: 18,22


.. exercise:: Exercise03: ``TEAMS`` vs  ``PARALLEL`` constructs

   We start from the "hello world" example, and by adding ``TEAMS`` and  ``PARALLEL`` constructs
   to compare the differences. Furthermore, using ``num_teams`` and ``thread_limit`` to limit
   the number of teams and threads to be generated.

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: exercise/ex03/ex03.c
            :language: c
            :linenos:


      .. tab:: Fortran

	 .. literalinclude:: exercise/ex03/ex03.F90                  
	    :language: fortran
            :linenos:
            

.. solution:: 

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: exercise/ex03/solution/ex03.c
            :language: c
            :linenos:
	    :emphasize-lines: 13,14


      .. tab:: Fortran

	 .. literalinclude:: exercise/ex03/solution/ex03.F90                  
	    :language: fortran
            :linenos:
            :emphasize-lines: 15,16,20,21



Composite directive
-------------------

It is convenient to use the composite construct

  - the code is more portable 
  - let the compiler figure out the loop tiling since each compiler
    supports different levels of parallelism
  - possible to reach good performance without composite directives


.. challenge:: Syntax

   .. tabs::

      .. tab:: C/C++

             .. code-block:: c

		  #pragma omp target teams distribute parallel for simd [clauses]
		  	for-loops
		  


      .. tab:: Fortran

             .. code-block:: fortran

		  !$omp target teams distribute parallel do simd [clauses]
		          do-loops
		  [!$omp end target teams distribute parallel do simd]




.. exercise:: Exercise: Offloading

   We will start from the serial version of the heat diffusion and step by step
   add the directives for offloading and parallelism on the target device.  Compare 
   the performance to understand the effects of different directives. We will 
   focus on the core evoluton operation only for now, i.e. subroutine evolve 
   in the file core.cpp or core.F90.  

   step 1: adding the ``TARGET`` construct 

   step 2: adding the ``TARGET TEAMS`` construct

   step 3: adding the ``TARGET TEAMS DISTRIBUTE`` construct

   step 4: adding the ``TARGET TEAMS DISTRIBUTE PARALLEL FOR/DO`` construct

   Use a small number of iterations, e.g. ./heat_serial 800 800 10, 
   otherwise it may take a long time to finish.

   The exercise is under /content/exercise/offloading

.. solution::

   .. tabs::

      .. tab:: C++

         .. literalinclude:: exercise/solution/offloading/core.cpp
                        :language: cpp
			:emphasize-lines: 24-25


      .. tab:: Fortran

         .. literalinclude:: exercise/solution/offloading/fortran/core.F90
                        :language: fortran
                        :emphasize-lines: 35,45


