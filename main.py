import os
import time

from fenics import (
    parameters,
    RectangleMesh,
    Point,
    BoundaryMesh,
    SubMesh,
    HDF5File,
    MPI,
    MeshFunction,
    cells,
    refine,
    VectorFunctionSpace,
    ALE,
    Expression,
    project,
)
from mshr import Rectangle, Circle, generate_mesh, Polygon
from parameters import Parameters
from spaces import Space
from time_structure import TimeLine, split
from time_stepping import time_stepping
from forms import (
    bilinear_form_structure,
    functional_structure,
    bilinear_form_thermal,
    functional_thermal,
    bilinear_form_structure_adjoint,
    functional_structure_adjoint,
    bilinear_form_thermal_adjoint,
    functional_thermal_adjoint,
    Problem,
)
from relaxation import relaxation
from shooting import shooting
from compute_residuals import compute_residuals
from refine import refine_time
from error_estimate import goal_functional_structure

parameters["allow_extrapolation"] = True
param = Parameters()

# Create meshes
structure_horizontal = Rectangle(Point(0.0, 0.0), Point(0.502, 0.02))
structure_vertical = Rectangle(Point(0.2, -0.2), Point(0.22, 0.02))
domain = structure_horizontal + structure_vertical
mesh = generate_mesh(domain, 35)

# Create function spaces
structure_dimension = [2, 2]
structure_degree = [1, 1]
thermal_dimension = [1]
thermal_degree = [1]
structure = Space(mesh, structure_dimension, structure_degree, "structure")
thermal = Space(mesh, thermal_dimension, thermal_degree, "thermal")

# Define variational forms
structure.primal_problem = Problem(
    bilinear_form_structure, functional_structure
)
structure.adjoint_problem = Problem(
    bilinear_form_structure_adjoint,
    functional_structure_adjoint,
)
thermal.primal_problem = Problem(
    bilinear_form_thermal,
    functional_thermal,
)
thermal.adjoint_problem = Problem(
    bilinear_form_thermal_adjoint,
    functional_thermal_adjoint,
)

# Create time interval structures
structure_timeline = TimeLine()
structure_timeline.unify(
    param.TIME_STEP,
    param.LOCAL_MESH_SIZE_STRUCTURE,
    param.GLOBAL_MESH_SIZE,
    param.INITIAL_TIME,
)
thermal_timeline = TimeLine()
thermal_timeline.unify(
    param.TIME_STEP,
    param.LOCAL_MESH_SIZE_THERMAL,
    param.GLOBAL_MESH_SIZE,
    param.INITIAL_TIME,
)

# Set deoupling method
if param.RELAXATION:

    decoupling = relaxation

else:

    decoupling = shooting

# Refine time meshes
structure_size = structure_timeline.size_global - structure_timeline.size
thermal_size = thermal_timeline.size_global - thermal_timeline.size
for i in range(param.REFINEMENT_LEVELS):

    structure_refinements_txt = open(
        f"structure_{structure_size}-{thermal_size}_refinements.txt", "r"
    )
    thermal_refinements_txt = open(
        f"thermal_{structure_size}-{thermal_size}_refinements.txt", "r"
    )
    structure_refinements = [
        bool(int(x)) for x in structure_refinements_txt.read()
    ]
    thermal_refinements = [
        bool(int(x)) for x in thermal_refinements_txt.read()
    ]
    structure_refinements_txt.close()
    thermal_refinements_txt.close()
    structure_timeline.refine(structure_refinements)
    thermal_timeline.refine(thermal_refinements)
    split(structure_timeline, thermal_timeline)
    structure_size = structure_timeline.size_global - structure_timeline.size
    thermal_size = thermal_timeline.size_global - thermal_timeline.size
    print(f"Global number of macro time-steps: {structure_timeline.size}")
    print(
        f"Global number of micro time-steps in the structure timeline: {structure_size}"
    )
    print(
        f"Global number of micro time-steps in the thermal timeline: {thermal_size}"
    )
structure_timeline.print()
thermal_timeline.print()

# Create directory
structure_size = structure_timeline.size_global - structure_timeline.size
thermal_size = thermal_timeline.size_global - thermal_timeline.size
try:

    os.makedirs(f"{structure_size}-{thermal_size}")

except FileExistsError:

    pass
os.chdir(f"{structure_size}-{thermal_size}")

# Perform time-stepping of the primal problem
adjoint = False
start = time.time()
if param.COMPUTE_PRIMAL:
    time_stepping(
        structure,
        thermal,
        param,
        decoupling,
        structure_timeline,
        thermal_timeline,
        adjoint,
    )
end = time.time()
# print(end - start)
structure_timeline.load(structure, "structure", adjoint)
thermal_timeline.load(thermal, "thermal", adjoint)

# Compute goal functional
goal_functional = goal_functional_structure(
    structure,
    structure_timeline,
    thermal,
    thermal_timeline,
    param,
)

print(f"Value of goal functional: {sum(goal_functional)}")

# Perform time-stepping of the adjoint problem
adjoint = True
if param.COMPUTE_ADJOINT:
    time_stepping(
        structure,
        thermal,
        param,
        decoupling,
        structure_timeline,
        thermal_timeline,
        adjoint,
    )
structure_timeline.load(structure, "structure", adjoint)
thermal_timeline.load(thermal, "thermal", adjoint)

# Compute residuals
(
    primal_residual_structure,
    primal_residual_thermal,
    adjoint_residual_structure,
    adjoint_residual_thermal,
    goal_functional,
) = compute_residuals(
    structure, thermal, param, structure_timeline, thermal_timeline
)

# Refine mesh
refine_time(
    primal_residual_structure,
    primal_residual_thermal,
    adjoint_residual_structure,
    adjoint_residual_thermal,
    structure_timeline,
    thermal_timeline,
)
