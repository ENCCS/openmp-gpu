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



.. challenge:: Exercise04: explicit and implicit data mapping 

   1. explicitly adding the ``map`` clauses for data transfer between the host and device 
   2. offloading the part where it "calculates the sum"

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: exercise/ex04/ex04.c
            :language: c
            :linenos:


      .. tab:: Fortran

	 .. literalinclude:: exercise/ex04/ex04.F90                  
	    :language: fortran
            :linenos:


.. solution:: 

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: exercise/ex04/solution/ex04.c
            :language: c
            :linenos:
	    :emphasize-lines: 17,24


      .. tab:: Fortran

	 .. literalinclude:: exercise/ex04/solution/ex04.F90                 
	    :language: fortran
            :linenos:
            :emphasize-lines: 18,26,30



Data region
-----------

How the ``TARGET`` construct creates storage, transfer data, and remove
storage on the device are clasiffied as two categories: structured
data region and unstructured data region.

Structured Data Regions
~~~~~~~~~~~~~~~~~~~~~~~

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



Unstructured Data Regions
~~~~~~~~~~~~~~~~~~~~~~~~~

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




TARGET UPDATE construct
~~~~~~~~~~~~~~~~~~~~~~~

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



.. challenge:: Exercise05: ``TARGET DATA`` structured region  

   Create a data region using ``TARGET DATA`` and add ``map`` clauses for data transfer.

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: exercise/ex05/ex05.c
            :language: c
            :linenos:


      .. tab:: Fortran

	 .. literalinclude:: exercise/ex05/ex05.F90                  
	    :language: fortran
            :linenos:

.. solution:: 

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: exercise/ex05/solution/ex05.c
            :language: c
            :linenos:
	    :emphasize-lines: 17,18,19,30,34


      .. tab:: Fortran

	 .. literalinclude:: exercise/ex05/solution/ex05.F90                 
	    :language: fortran
            :linenos:
            :emphasize-lines: 18,19,31,36



.. challenge:: Exercise06:  ``TARGET UPDATE``

   Trying to figure out the variable values on host and device at each check point.

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: exercise/ex06/ex06.c
            :language: c
            :linenos:


      .. tab:: Fortran

	 .. literalinclude:: exercise/ex06/ex06.F90                  
	    :language: fortran
            :linenos:

    
      .. tab:: Solution

		+-------------+---------+-----------+
		|check point  |x on host|x on device|
		+=============+=========+===========+
		|check point1 |   0     |  0        | 
		+-------------+---------+-----------+
		|check point2 |  10     |  0        | 
		+-------------+---------+-----------+
		|check point3 |  10     | 10        | 
	        +-------------+---------+-----------+

   



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



