from fenics import (
    near,
    SubDomain,
    CellDiameter,
    FacetNormal,
    Measure,
    MeshFunction,
    FiniteElement,
    VectorElement,
    FunctionSpace,
    Constant,
    Expression,
    DirichletBC,
    between,
    MixedElement,
)

# Define boundary parts
def bottom(x, on_boundary):
    return on_boundary and between(x[0], (0.2, 0.22)) and near(x[1], -0.2)


def boundary(x, on_boundary):
    return on_boundary


# Store space attributes
class Space:
    def __init__(self, mesh, dimension, degree, name):

        # Define mesh parameters
        self.mesh = mesh
        self.cell_size = CellDiameter(mesh)
        self.normal_vector = FacetNormal(mesh)

        # Define measures
        self.dx = Measure("dx", domain=mesh)
        self.ds = Measure("ds", domain=mesh)

        # Define function spaces
        finite_element = []
        if name == "structure":
            number_of_variables = 2
        else:
            number_of_variables = 1
        for i in range(number_of_variables):
            if dimension[i] > 1:
                finite_element.append(
                    VectorElement("CG", mesh.ufl_cell(), degree[i])
                )
            else:
                finite_element.append(
                    FiniteElement("CG", mesh.ufl_cell(), degree[i])
                )
        if name == "structure":
            self.function_space = FunctionSpace(
                mesh,
                MixedElement([finite_element[0], finite_element[1]]),
            )
            self.function_space_split = [
                self.function_space.sub(0).collapse(),
                self.function_space.sub(1).collapse(),
            ]
        else:
            self.function_space = FunctionSpace(mesh, "CG", 1)
            self.function_space_split = [FunctionSpace(mesh, "CG", 1)]

        # Define additional information
        self.dimension = dimension
        self.degree = degree
        self.name = name

    # Define boundaries
    def boundaries(self):

        if self.name == "structure":
            return [
                DirichletBC(
                    self.function_space.sub(0),
                    Constant((0.0, 0.0)),
                    bottom,
                ),
                DirichletBC(
                    self.function_space.sub(1),
                    Constant((0.0, 0.0)),
                    bottom,
                ),
            ]
        else:
            return
