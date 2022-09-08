A* Examples
===============

.. contents:: Examples
   :depth: 2
   :local:


Normal A*
---------
The A* algorithm can be run with the following.

.. literalinclude:: /examples/planning/a_star.py
   :linenos:
   :pyobject: normal_a_star

which gives this as output.

.. image:: /examples/planning/a_star_normal.gif
   :align: center


Beam Search
-----------
A modified A* algorithm called beam search can be run with the following.

.. literalinclude:: /examples/planning/a_star.py
   :linenos:
   :pyobject: beam_search

which gives this as output.

.. image:: /examples/planning/a_star_beam_search.gif
   :align: center


Weighted A*
-----------
A modified A* algorithm that weights the heuristic can be run with the following.

.. literalinclude:: /examples/planning/a_star.py
   :linenos:
   :pyobject: weighted_a_star

which gives this as output.

.. image:: /examples/planning/a_star_weighted.gif
   :align: center
