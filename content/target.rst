Offloading to GPU
=================

.. prereq::

   more to add:

   1. target update/ target declare
   2. excercise
   3. loop directive from v5.0 


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
  - one or multiple target devices *of the same kind*: e.g. CPU, GPU, FPGA, ...
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
  5.The host destroys the data environment on the device.



TARGET construct
----------------

The ``TARGET`` construct consists of a *target* directive and an
execution region. It is used to transfer both the control flow from
the host to the device and the data between the host and device.


.. typealong:: Syntax

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: syntax/v4.5.0/target.c
                        :language: c

         .. literalinclude:: syntax/v4.5.0/target.clause
                        :language: c

      .. tab:: Fortran

         .. literalinclude:: syntax/v4.5.0/target.f90
                        :language: fortran

         .. literalinclude:: syntax/v4.5.0/target.clause
                        :language: fortran




Clauses:

  - device(scalar-integer-expression)
  - if(scalar-expression)
    - If the scalar-expression evaluates to false then the target region is executed by the host device in the host data environment.
  - device(integer-expression)
    – The value of the integer-expression selects the device when a device other than the default device is desired.
  - private(list) firstprivate(list)
    – creates variables with the same name as those in the list on the device. In the case of firstprivate, the value of the variable on the host is copied into the private variable created on the device.
  - map([map-type:] list)
    – map-type may be to, from, tofrom, or alloc. The clause defines how the variables in list are moved between the host and the device. 
  - nowait
    – The target task is deferred which means the host can run code in parallel to the target region on the device.




.. challenge:: Example: target construct 

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: examples/v4.5.0/Example_target.1.c
                        :language: c

      .. tab:: Fortran

         .. literalinclude:: examples/v4.5.0/Example_target.1.f90
                        :language: fortran



Creating Parallelism on the Target Device
-----------------------------------------

The target construct transfers the control flow to the device is
sequential and synchronous, and it is because OpenMP separates offload
and parallelism.  One needs to explicitly create parallel regions on
the target device to make efficient use of the device(s).

Teams construct
---------------

.. typealong:: Syntax

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: syntax/v4.5.0/teams.c
                        :language: c

         .. literalinclude:: syntax/v4.5.0/teams.clause
                        :language: c

      .. tab:: Fortran

         .. literalinclude:: syntax/v4.5.0/teams.f90
                        :language: fortran

         .. literalinclude:: syntax/v4.5.0/teams.clause
                        :language: fortran


The teams construct spawns a league of teams.  The maximum number of
teams is specified by the num_teams clause, Each team executes with
thread_limit threads Each team in the league starts with one master
thread and *concurrent (not parallel) execution* on each Streaming
Multiprocessors Threads in a team can synchronize but no
synchronization among teams The construct must be “perfectly” nested
in a target construct



Distribute construct
--------------------

.. typealong:: Syntax

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: syntax/v4.5.0/distribute.c
                        :language: c

         .. literalinclude:: syntax/v4.5.0/distribute.clause
                        :language: c

      .. tab:: Fortran

         .. literalinclude:: syntax/v4.5.0/distribute.f90
                        :language: fortran

         .. literalinclude:: syntax/v4.5.0/distribute.clause
                        :language: fortran


Loop iterations are workshared across the master threads in the teams,
but no worksharing within the threads in one team No implicit barrier
at the end of the construct no guarantee about the order the teams
will execute.

.. challenge:: Example: teams and distribute constructs 

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: examples/v4.5.0/Example_teams.6.c
                        :language: c

      .. tab:: Fortran

         .. literalinclude:: examples/v4.5.0/Example_teams.6.f90
         		:language: fortran   



Composite directive
-------------------

It is convenient to use the composite construct

  - the code is more portable 
  - let the compiler figure out the loop tiling since each compiler
    supports different levels of parallelism
  - possible to reach good performance without composite directives



.. typealong:: Syntax

   .. tabs::

      .. tab:: C/C++

         .. literalinclude:: syntax/v4.5.0/composite.c
                        :language: c

      .. tab:: Fortran

         .. literalinclude:: syntax/v4.5.0/composite.f90
                        :language: fortran


