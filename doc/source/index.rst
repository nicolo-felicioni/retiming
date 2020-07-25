.. Retiming documentation master file, created by
   sphinx-quickstart on Thu Jul 23 12:48:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Retiming's documentation!
====================================
The implementation relies on the `NetworkX (link: https://networkx.github.io)`
library to build graphs, and uses the wrapper pattern to wrap the DiGraph object of NetworkX
(for directed graph) within another object that implements the algorithm to be implemented and
other auxiliary methods.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   GraphWrapper <GraphWrapper.rst>

   NewGraphWrapper <NewGraphWrapper.rst>

   test_generator <test_generator.rst>

   utils <utils.rst>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
