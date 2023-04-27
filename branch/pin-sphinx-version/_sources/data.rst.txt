Data environment
================

.. objectives::

   - Understand what are data movement
   - Understand what are structured and unstructured data clauses
   - Learn how to move data explicitly
   - Learn how to compile and run OpenMP code with data movement directives

.. prereq::

   1. You need ...
   2. Basic understanding of ...

Data mapping
------------

Due to distinct memory spaces on host and device, transferring data
becomes inevitable. The map cluase on a device construct explicitly
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
   
.. note::

	When mapping data arrays or pointers, be careful about the array section notation:
	  - In C/C++: array[lower-bound:length]
	  - In Fortran:array[lower-bound:upper-bound]
	



Data region
-----------

How the target construct creates storage, transfer data, and remove
storage on the device are clasiffied as two categories: structured
data region and unstructured data region.

Structured Data Regions
-----------------------

The target data construct is used to create a structured data region
which is convenient for providing persistent data on the device which
could be used for subseqent target constructs.

.. typealong:: Syntax

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: syntax/v4.5.0/target_data.c
                        :language: c

         .. literalinclude:: syntax/v4.5.0/target_data.clause
                        :language: c

      .. tab:: Fortran

         .. literalinclude:: syntax/v4.5.0/target_data.f90
                        :language: fortran

         .. literalinclude:: syntax/v4.5.0/target_data.clause
                        :language: fortran




.. challenge:: Example:  target structured data 

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: examples/v4.5.0/Example_target_data.2.c
                        :language: c

      .. tab:: Fortran

         .. literalinclude:: examples/v4.5.0/Example_target_data.2.f90
         		:language: fortran   


Unstructured Data Regions
-------------------------

The unstructured data constructs have much more freedom in creating
and deleting of data on the device at any appropriate point.

.. typealong:: Syntax

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: syntax/v4.5.0/target_enter_data.c
                        :language: c

         .. literalinclude:: syntax/v4.5.0/target_exit_data.c
                        :language: c

         .. literalinclude:: syntax/v4.5.0/target_enter_exit_data.clause
                        :language: c

      .. tab:: Fortran

         .. literalinclude:: syntax/v4.5.0/target_enter_data.f90
                        :language: fortran

         .. literalinclude:: syntax/v4.5.0/target_exit_data.f90
                        :language: fortran

         .. literalinclude:: syntax/v4.5.0/target_enter_exit_data.clause
                        :language: fortran


.. challenge:: Example:  target unstructured data

	The unstructured data constructs have much more freedom in
	creating and deleting of data on the device at any appropriate
	point.

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: examples/v4.5.0/Example_target_unstructured_data.1.c
                        :language: c

      .. tab:: Fortran

         .. literalinclude:: examples/v4.5.0/Example_target_unstructured_data.1.f90
         		:language: fortran   


Optimize Data Transfers
-----------------------

- Reduce the amount of data mapping between host and device
- Try to keep data environment residing on the target device as long
  as possible

