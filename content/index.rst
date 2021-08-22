OpenMP for GPU offloading
=========================

This course is intended for newcomers to OpenMP GPU offloading.
By the end of it, students will feel comfortable with the basic process of introducing OpenMP offloading constructs to a simple code base.
They will then be able to

   - reason about which parts of the code to change,
   - know how to manage data transfers, lifetimes and reductions,
   - use a mini-app to observe behavior
   - be aware of how to interact with GPU libraries
   - be aware of how to interact with MPI

.. prereq::

   - Understanding of how to read either C++ or Fortran code
   - Running programs from a terminal command line
   - For one lesson, familiarity with MPI concepts for multi-node programming



.. csv-table::
   :widths: auto
   :delim: ;

   40 min ; :doc:`miniapp`
   20 min ; :doc:`target`
   20 min ; :doc:`data`

.. toctree::
   :maxdepth: 1
   :caption: The lessons

   miniapp
   target
   data

.. toctree::
   :maxdepth: 1
   :caption: Reference

   quick-reference
   guide



.. _learner-personas:

Who is the course for?
----------------------

This course is for all people who seek to understand the basics of how OpenMP offloading for GPUs can improve the performance of their code.



About the course
----------------

This course is offered jointly by ENCCS (https://enccs.se/) and CSC (https://www.csc.fi/)



See also
--------

https://github.com/OpenMP/Examples
https://github.com/UoB-HPC/openmp-tutorial


Credits
-------

The lesson file structure and browsing layout is inspired by and derived from
`work <https://github.com/coderefinery/sphinx-lesson>`_ by `CodeRefinery
<https://coderefinery.org/>`_ licensed under the `MIT license`_.
We have copied and adapted most of their license text.

Instructional Material
^^^^^^^^^^^^^^^^^^^^^^

This instructional material is made available under the
`Creative Commons Attribution license (CC-BY-4.0) <https://creativecommons.org/licenses/by/4.0/>`_.
The following is a human-readable summary of (and not a substitute for) the
`full legal text of the CC-BY-4.0 license
<https://creativecommons.org/licenses/by/4.0/legalcode>`_.
You are free to:

- **share** - copy and redistribute the material in any medium or format
- **adapt** - remix, transform, and build upon the material for any purpose,
  even commercially.

The licensor cannot revoke these freedoms as long as you follow these license terms:

- **Attribution** - You must give appropriate credit (mentioning that your work
  is derived from work that is Copyright (c) Thor Wikfeldt and individual contributors and, where practical, linking
  to `<https://enccs.se>`_), provide a `link to the license
  <https://creativecommons.org/licenses/by/4.0/>`_, and indicate if changes were
  made. You may do so in any reasonable manner, but not in any way that suggests
  the licensor endorses you or your use.
- **No additional restrictions** - You may not apply legal terms or
  technological measures that legally restrict others from doing anything the
  license permits.

With the understanding that:

- You do not have to comply with the license for elements of the material in
  the public domain or where your use is permitted by an applicable exception
  or limitation.
- No warranties are given. The license may not give you all of the permissions
  necessary for your intended use. For example, other rights such as
  publicity, privacy, or moral rights may limit how you use the material.



Software
^^^^^^^^

Except where otherwise noted, the example programs and other software provided
with this repository are made available under the `OSI <http://opensource.org/>`_-approved
`MIT license`_.


.. _MIT license: https://opensource.org/licenses/mit-license.html
