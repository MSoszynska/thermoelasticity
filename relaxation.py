from fenics import Function, FunctionSpace, project, interpolate, norm
from solve_structure import solve_structure
from solve_thermal import solve_thermal
from parameters import Parameters
from spaces import Space
from time_structure import MacroTimeStep
from initial import Initial

# Define relaxation method
def relaxation(
    velocity: Initial,
    displacement: Initial,
    temperature: Initial,
    first_time_step,
    structure: Space,
    thermal: Space,
    param: Parameters,
    macrotimestep_structure: MacroTimeStep,
    macrotimestep_thermal: MacroTimeStep,
    adjoint,
    counter_structure,
    counter_thermal,
    primal_solutions,
):

    # Define initial values for relaxation method
    temperature_new = Function(thermal.function_space_split[0])
    number_of_iterations = 0
    stop = False

    while not stop:

        number_of_iterations += 1
        print(
            f"Current iteration of relaxation method: {number_of_iterations}"
        )

        # Save old values
        temperature_new.assign(temperature.new)

        # Perform one iteration
        solve_structure(
            velocity,
            displacement,
            temperature,
            structure,
            thermal,
            first_time_step,
            param,
            macrotimestep_structure,
            macrotimestep_thermal,
            adjoint,
            counter_structure,
            primal_solutions,
        )
        solve_thermal(
            velocity,
            displacement,
            temperature,
            structure,
            thermal,
            first_time_step,
            param,
            macrotimestep_structure,
            macrotimestep_thermal,
            adjoint,
            counter_thermal,
            primal_solutions,
        )

        # Perform relaxation
        temperature.new.assign(
            project(
                param.TAU * temperature.new
                + (1.0 - param.TAU) * temperature_new,
                thermal.function_space_split[0],
            )
        )

        # Define errors on the interface
        velocity_error = project(
            temperature_new - temperature.new,
            thermal.function_space_split[0],
        )
        error_linf = norm(velocity_error.vector(), "linf")
        if number_of_iterations == 1:

            if error_linf != 0.0:
                error_initial_linf = error_linf
            else:
                error_initial_linf = 1.0

        print(f"Absolute error on the interface: {error_linf}")
        print(
            f"Relative error on the interface: {error_linf / error_initial_linf}"
        )

        # Check stop conditions
        if (
            error_linf < param.ABSOLUTE_TOLERANCE_RELAXATION
            or error_linf / error_initial_linf
            < param.RELATIVE_TOLERANCE_RELAXATION
        ):

            print(
                f"Algorithm converged successfully after "
                f"{number_of_iterations} iterations"
            )
            stop = True

        elif number_of_iterations == param.MAX_ITERATIONS_RELAXATION:

            print("Maximal number of iterations was reached.")
            stop = True
            number_of_iterations = -1

    velocity.iterations.append(number_of_iterations)
    displacement.iterations.append(number_of_iterations)
    temperature.iterations.append(number_of_iterations)

    return
