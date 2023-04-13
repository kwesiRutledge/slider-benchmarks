"""
test_scipy_minimize.py
Description:
    In this script, we test scipy minimize's ability to solve the convex combination decomposition problem.
    Can it uniquely find the convex combination weights for a given polytope which can be used to define a unique point
    in the polytope?
"""

import numpy
import scipy.optimize as optimize
import polytope as pc

# Constants
A = numpy.array([
    [1, 0],
    [0, -1],
    [-1, 1],
])
b = numpy.array([0, 0, 1])

poly1 = pc.Polytope(A, b)
poly1_vertices = pc.extreme(poly1)
print(poly1_vertices)

target_point1 = numpy.array([-0.1, 0.1])

input_poly = pc.box2poly([[0, 1], [0, 1], [0, 1]])

# Define the objective function
obj = lambda theta: numpy.linalg.norm(poly1_vertices.T @ theta - target_point1)
# Define constraint
tempConstr = optimize.LinearConstraint(numpy.eye(3), numpy.zeros((3,)), numpy.ones((3,)))
print(tempConstr.A, tempConstr.lb, tempConstr.ub)


print("\n\nUsing optimize.minimize")
print(
    optimize.minimize(
        obj,
        x0=numpy.array([0.5, 0.25, 0.25]),
        constraints=(tempConstr),
    )
)


print("\n\nSolve with the \'preferred\' method: optimize.lsq_linear")
print(
    optimize.lsq_linear(poly1_vertices.T, target_point1, (0, 1))
)