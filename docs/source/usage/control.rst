Control Examples
================

.. contents:: Examples
   :depth: 2
   :local:


LQR Infinite Horizon Linear Dyanmics
------------------------------------
The LQR algorithm can be run for the infinite horizon case using linear dynamics
with the following. Note the state trajectory output can be used as a motion plan.

.. literalinclude:: /examples/control/lqr.py
   :linenos:
   :pyobject: linear_inf_horizon

which gives this as output.

.. image:: /examples/control/lqr_linear_inf_horizon.png
   :align: center


LQR Finite Horzon Linear Dynamics
---------------------------------
The LQR algorithm can be run for the finite horizon case using non-linear dynamics
with the following. Note the state trajectory output can be used as a motion plan.

.. literalinclude:: /examples/control/lqr.py
   :linenos:
   :pyobject: linear_finite_horizon

which gives this as output.

.. image:: /examples/control/lqr_linear_finite_horizon.png
   :align: center


LQR Finite Horzon Non-linear Dynamics
-------------------------------------
The LQR algorithm can be run for the finite horizon case using non-linear dynamics
with the following. Note the state trajectory output can be used as a motion plan.

.. literalinclude:: /examples/control/lqr.py
   :linenos:
   :pyobject: nonlin_finite_hor

which gives this as output.

.. image:: /examples/control/lqr_nonlinear_finite_horizon.png
   :align: center


ELQR
----
The ELQR algorithm can be run for a finite horizon with non-linear dynamics and
a non-quadratic cost with the following. Note the state trajectory output can be
used as a motion plan. This version does not modify the quadratization process.

.. literalinclude:: /examples/control/elqr.py
   :linenos:
   :pyobject: basic

which gives this as output.

.. image:: /examples/control/elqr_basic.gif
   :align: center


ELQR Modified Quadratization
----------------------------
The ELQR algorithm can be run for a finite horizon with non-linear dynamics and
a non-quadratic cost with the following. Note the state trajectory output can be
used as a motion plan. This version does modifies the quadratization process.

.. literalinclude:: /examples/control/elqr.py
   :linenos:
   :pyobject: modify_quadratize

which gives this as output.

.. image:: /examples/control/elqr_modify_quadratize.gif
   :align: center
