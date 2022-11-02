import numpy as np
from fenics import (
    Function,
    FunctionSpace,
    interpolate,
    project,
    inner,
    Constant,
    DirichletBC,
    assign,
    dot,
    Expression,
    det,
    inv,
    assemble,
)
from solve_structure import solve_structure
from solve_thermal import solve_thermal
from scipy.sparse.linalg import LinearOperator, gmres
from parameters import Parameters
from spaces import Space
from initial import Initial
from time_structure import MacroTimeStep

# Define shooting function
def shooting_function(
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

    # Save old values
    temperature_new = Function(thermal.function_space_split[0])
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

    # Define an update vector
    temperature_update = project(
        temperature_new - temperature.new,
        thermal.function_space_split[0],
    )

    # Represent shooting function as an array
    temperature_update_array = temperature_update.vector().get_local()

    return temperature_update_array


# Define linear operator for linear solver in shooting method
def shooting_newton(
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
    direction_old,
    shooting_function_value,
):
    def shooting_gmres(direction):

        # Define empty increment temperature
        increment_temperature = Function(thermal.function_space_split[0])

        # Set values of increment temperature
        increment_temperature.vector().set_local(
            direction_old + param.EPSILON * direction
        )
        temperature.new.assign(increment_temperature)

        # Compute shooting function
        shooting_function_increment = shooting_function(
            velocity,
            displacement,
            temperature,
            first_time_step,
            structure,
            thermal,
            param,
            macrotimestep_structure,
            macrotimestep_thermal,
            adjoint,
            counter_structure,
            counter_thermal,
            primal_solutions,
        )

        return (
            shooting_function_increment - shooting_function_value
        ) / param.EPSILON

    return shooting_gmres


def shooting(
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

    # Define initial values for Newton's method
    if macrotimestep_thermal.before is not None:
        macrotimestep_thermal_before_dt = macrotimestep_thermal.before.dt
    else:
        macrotimestep_thermal_before_dt = macrotimestep_thermal.dt
    temperature_new = Function(thermal.function_space_split[0])
    # temperature_new.assign(
    #     project(
    #         temperature.old
    #         + macrotimestep_thermal.dt
    #         / macrotimestep_thermal_before_dt
    #         * (temperature.old - temperature.old_old),
    #         thermal.function_space_split[0],
    #     )
    # )
    temperature_new.assign(temperature.new)
    number_of_iterations = 0
    number_of_linear_systems = 0
    stop = False

    # Define Newton's method
    while not stop:

        number_of_iterations += 1
        number_of_linear_systems += 1
        print(f"Current iteration of Newton's method: {number_of_iterations}")

        # Define right hand side
        temperature.new.assign(temperature_new)
        shooting_function_value = shooting_function(
            velocity,
            displacement,
            temperature,
            first_time_step,
            structure,
            thermal,
            param,
            macrotimestep_structure,
            macrotimestep_thermal,
            adjoint,
            counter_structure,
            counter_thermal,
            primal_solutions,
        )
        shooting_function_value_linf = np.max(np.abs(shooting_function_value))
        if number_of_iterations == 1:

            if shooting_function_value_linf != 0.0:
                shooting_function_value_initial_linf = (
                    shooting_function_value_linf
                )
            else:
                shooting_function_value_initial_linf = 1.0

        print(
            f"Absolute error on the interface in infinity norm: "
            f"{shooting_function_value_linf}"
        )
        print(
            f"Relative error on the interface in infinity norm: "
            f"{shooting_function_value_linf / shooting_function_value_initial_linf}"
        )

        # Check stop conditions
        if (
            shooting_function_value_linf < param.ABSOLUTE_TOLERANCE_NEWTON
            or shooting_function_value_linf
            / shooting_function_value_initial_linf
            < param.RELATIVE_TOLERANCE_NEWTON
        ):
            print(
                f"Newton's method converged successfully after "
                f"{number_of_iterations} iterations and solving "
                f"{number_of_linear_systems} linear systems."
            )
            stop = True

        elif number_of_iterations == param.MAX_ITERATIONS_NEWTON:

            print("Newton's method failed to converge.")
            stop = True
            number_of_linear_systems = -1

        if not stop:

            # Define linear operator
            temperature_array = temperature_new.vector().get_local()
            linear_operator_newton = shooting_newton(
                velocity,
                displacement,
                temperature,
                first_time_step,
                structure,
                thermal,
                param,
                macrotimestep_structure,
                macrotimestep_thermal,
                adjoint,
                counter_structure,
                counter_thermal,
                primal_solutions,
                temperature_array,
                shooting_function_value,
            )
            operator_size = len(temperature_array)
            shooting_gmres = LinearOperator(
                (operator_size, operator_size), matvec=linear_operator_newton
            )

            # Solve linear system
            number_of_iterations_gmres = 0

            def callback(vector):

                nonlocal number_of_iterations_gmres
                global residual_norm_gmres
                number_of_iterations_gmres += 1
                print(
                    f"Current iteration of GMRES method: {number_of_iterations_gmres}"
                )
                residual_norm_gmres = np.linalg.norm(vector)

            direction, exit_code = gmres(
                shooting_gmres,
                -shooting_function_value,
                tol=param.TOLERANCE_GMRES,
                restart=30,
                maxiter=param.MAX_ITERATIONS_GMRES,
                callback=callback,
            )
            number_of_linear_systems += number_of_iterations_gmres
            if exit_code == 0:

                print(
                    f"GMRES method converged successfully after "
                    f"{number_of_iterations_gmres} iterations"
                )

            else:

                print("GMRES method failed to converge.")
                print(f"Norm of residual: {residual_norm_gmres}")

            # Advance solution
            temperature_array += direction
            temperature_new.vector().set_local(temperature_array)

    velocity.iterations.append(number_of_linear_systems)
    displacement.iterations.append(number_of_linear_systems)
    temperature.iterations.append(number_of_linear_systems)

    return
