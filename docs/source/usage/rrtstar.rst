RRT* Examples
=============

.. contents:: Examples
   :depth: 2
   :local:


LQR-RRT* with Linear Dynamics
-----------------------------
The LQR-RRT* algorithm can be run with linear dynamics using the following.

.. literalinclude:: /examples/planning/rrt_star.py
   :linenos:
   :pyobject: lin_lqrrrtstar

which gives this as output.

.. image:: /examples/planning/lqrrrtstar_linear.gif
   :align: center


LQR-RRT* with Non-Linear Dynamics
---------------------------------
The LQR-RRT* algorithm can be run with non-linear dynamics using the following.

.. literalinclude:: /examples/planning/rrt_star.py
   :linenos:
   :pyobject: nonlin_lqrrrtstar

which gives this as output.

.. image:: /examples/planning/lqrrrtstar_nonlinear.gif
   :align: center
