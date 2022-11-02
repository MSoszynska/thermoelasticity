from fenics import Constant

# Define parameters
class Parameters:
    def __init__(
        self,
        rho=Constant(7800.0),
        nu=Constant(0.29),
        e_initial=Constant(210000000000),
        kappa=Constant(45.0),
        alpha=Constant(20000000000.0),
        beta=Constant(0.25),
        delta=Constant(0.01),
        c=Constant(480.0),
        initial_time=0.0,
        time_step=0.002,
        theta=0.5,
        global_mesh_size=50,
        local_mesh_size_structure=1,
        local_mesh_size_thermal=1,
        tau=Constant(0.5),
        absolute_tolerance_relaxation=1.0e-10,
        relative_tolerance_relaxation=1.0e-8,
        max_iterations_relaxation=100,
        epsilon=1.0e-6,
        absolute_tolerance_newton=1.0e-10,
        relative_tolerance_newton=1.0e-8,
        max_iterations_newton=10,
        tolerance_gmres=1.0e-10,
        max_iterations_gmres=25,
        relaxation=False,
        shooting=True,
        goal_functional_structure=True,
        goal_functional_thermal=False,
        compute_primal=True,
        compute_adjoint=True,
        refinement_levels=0,
        partial_load=0,
        partial_compute=None,  # [starting_point, size]
        initial_counter=0,
        adjoint_test_epsilon=0.0,  # * sqrt(10),
    ):

        # Define problem parameters
        self.RHO = rho
        self.NU = nu
        self.E_INITIAL = e_initial
        self.KAPPA = kappa
        self.ALPHA = alpha
        self.BETA = beta
        self.DELTA = delta
        self.C = c

        # Define time step on the coarsest level
        self.INITIAL_TIME = initial_time
        self.TIME_STEP = time_step
        self.THETA = theta

        # Define number of macro time steps on the coarsest level
        self.GLOBAL_MESH_SIZE = global_mesh_size

        # Define number of micro time-steps for structure
        self.LOCAL_MESH_SIZE_STRUCTURE = local_mesh_size_structure

        # Define number of micro time-steps for thermal
        self.LOCAL_MESH_SIZE_THERMAL = local_mesh_size_thermal

        # Define relaxation parameters
        self.TAU = tau
        self.ABSOLUTE_TOLERANCE_RELAXATION = absolute_tolerance_relaxation
        self.RELATIVE_TOLERANCE_RELAXATION = relative_tolerance_relaxation
        self.MAX_ITERATIONS_RELAXATION = max_iterations_relaxation

        # Define parameters for Newton's method
        self.EPSILON = epsilon
        self.ABSOLUTE_TOLERANCE_NEWTON = absolute_tolerance_newton
        self.RELATIVE_TOLERANCE_NEWTON = relative_tolerance_newton
        self.MAX_ITERATIONS_NEWTON = max_iterations_newton

        # Define parameters for GMRES method
        self.TOLERANCE_GMRES = tolerance_gmres
        self.MAX_ITERATIONS_GMRES = max_iterations_gmres

        # Choose decoupling method
        self.RELAXATION = relaxation
        self.SHOOTING = shooting

        # Choose goal functional
        self.GOAL_FUNCTIONAL_STRUCTURE = goal_functional_structure
        self.GOAL_FUNCTIONAL_THERMAL = goal_functional_thermal

        # Decide if primal and adjoint problems should be solved
        self.COMPUTE_PRIMAL = compute_primal
        self.COMPUTE_ADJOINT = compute_adjoint

        # Set number of refinement levels
        self.REFINEMENT_LEVELS = refinement_levels

        # Set the level of partial loading or computing of solutions
        self.PARTIAL_LOAD = partial_load
        self.PARTIAL_COMPUTE = partial_compute
        self.INITIAL_COUNTER = initial_counter

        # Set epsilon to test the adjoint
        self.ADJOINT_TEST_EPSILON = adjoint_test_epsilon
