Why OpenMP offloading?
======================

.. questions::

   - When and why should I use OpenMP offloading in my code?

.. objectives::

   - Understand shared parallel model
   - Understand program execution model
   - Understand basic constructs 

.. prereq::

   1. Basic C or FORTRAN


Computing in parallel
~~~~~~~~~~~~~~~~~~~~~

The underlying idea of parallel computing is to split a computational problem into smaller subtasks. Many subpsubtasks can then be solved *simultaneously* by multiple processing units. 

.. figure:: img/compp.png
   :align: center
   
   Computing in parallel.

How a problem is split into smaller subtasks depends fully on the problem. There are various paradigms and programming approaches how to do this. 

Distributed- vs. Shared-Memory Architecture
-------------------------------------------

Most of computing problems are not trivially parallelizable, which means that the subtasks need to have access from time to time to some of the results computed by other subtasks. The way subtasks exchange needed information depends on the available hardware.

.. figure:: img/distributed_vs_shared.png
   :align: center
   
   Distributed- vs shared-memory parallel computing.

In a distributed memory environment each computing unit operates independently from the others. It has its own memory and it  **cannot** access the memory in other nodes. The communication is done via network and each computing unit runs a separate copy of the operating system. In a shared memory machine all computing units have access to the memory and can read or modify the variables within.

Processes and threads
---------------------

The type of environment (distributed- or shared-memory) determines the programming model. There are two types of parallelism possible, process based and thread based. 

.. figure:: img/processes-threads.svg
   :align: center

For distributed memory machines a process basedparallel programming model is employed. The processes are independent execution units which have their *own memory* address spaces. They are created when the parallel program is started and they are only killed at the end. The communication between them is done explicitly via message passing like the MPI.

On the shared memoroy architectures it is possible to use a thread based parallelism.  The threads are light execution units and can be created and destryed at a relatively small cost. The threads have their own state information but they *share* the *same memory* adress space. When needed the communication is done though the shared memory. 

OpenMP
~~~~~~

Second heading
--------------

Some more text, with a figure

.. figure:: img/stencil.svg
   :align: center

   This is a sample image

.. exercise::

   TODO get the students to think about the content and answer a Zoom quiz

.. solution::

   Hide the answer and reasoning in here

Some source code
----------------

Sometimes we need to look at code, which can be in the webpage and optionally
you can pull out only some lines, or highlight others. Make sure both C++ and Fortran examples exist and work.

.. typealong:: The field data structure

   .. tabs::

      .. tab:: C++

         .. literalinclude:: code-samples/serial/heat.h
                        :language: cpp
                        :lines: 7-17
                                
      .. tab:: Fortran

         .. literalinclude:: code-samples/serial/fortran/heat_mod.F90
                        :language: fortran
                        :lines: 9-15

Building the code
-----------------

If there's terminal output to discuss, show something like::

  nvc++ -g -O3 -fopenmp -Wall -I../common -c main.cpp -o main.o
  nvc++ -g -O3 -fopenmp -Wall -I../common -c core.cpp -o core.o
  nvc++ -g -O3 -fopenmp -Wall -I../common -c setup.cpp -o setup.o
  nvc++ -g -O3 -fopenmp -Wall -I../common -c utilities.cpp -o utilities.o
  nvc++ -g -O3 -fopenmp -Wall -I../common -c io.cpp -o io.o
  nvc++ -g -O3 -fopenmp -Wall -I../common main.o core.o setup.o utilities.o io.o ../common/pngwriter.o -o heat_serial  -lpng


Running the code
----------------

To show a sample command line, use this approach

.. code-block:: bash

   ./heat_serial 800 800 1000


.. keypoints::

   - TODO summarize the learning outcome
   - TODO
