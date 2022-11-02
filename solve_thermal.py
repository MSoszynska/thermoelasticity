from fenics import (
    Function,
    FunctionSpace,
    project,
    DirichletBC,
    Constant,
    TrialFunction,
    split,
    TestFunction,
    solve,
    assemble,
    Expression,
    assign,
    action,
    derivative,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    info,
    assemble_system,
    norm,
    dot,
    inv,
)
from spaces import Space
from parameters import Parameters
from time_structure import MacroTimeStep
from initial import Initial


def solve_thermal(
    velocity: Initial,
    displacement: Initial,
    temperature: Initial,
    structure: Space,
    thermal: Space,
    first_time_step,
    param: Parameters,
    macrotimestep_structure: MacroTimeStep,
    macrotimestep_thermal: MacroTimeStep,
    adjoint,
    counter,
    primal_solutions,
    save=False,
):

    # Store old thermal solutions
    temperature_old = Function(thermal.function_space_split[0])
    temperature_old.assign(temperature.old)

    # Store old structurwe solutions
    velocity_old = Function(structure.function_space_split[0])
    displacement_old = Function(structure.function_space_split[1])
    velocity_old.assign(velocity.old)
    displacement_old.assign(displacement.old)

    # Initialize new structure solutions
    velocity_new = Function(structure.function_space_split[0])
    displacement_new = Function(structure.function_space_split[1])

    # Initialize primal solutions for the adjoint problem
    if adjoint:
        velocity_primal_new = Function(structure.function_space_split[0])
        displacement_primal_new = Function(structure.function_space_split[1])
        velocity_primal_old = Function(structure.function_space_split[0])
        displacement_primal_old = Function(structure.function_space_split[1])
        velocity_primal_old.assign(
            macrotimestep_structure.tail.functions["primal_velocity"]
        )
        displacement_primal_old.assign(
            macrotimestep_structure.tail.functions["primal_displacement"]
        )

    # Define time pointers
    if adjoint:

        microtimestep = macrotimestep_thermal.tail.before

    else:

        microtimestep = macrotimestep_thermal.head

    # Compute macro time-step size
    size = macrotimestep_thermal.size - 1
    local_counter = counter
    for m in range(size):

        # Extrapolate weak boundary conditions on the interface
        if adjoint:

            extrapolation_proportion = (
                microtimestep.point - macrotimestep_thermal.head.point
            ) / macrotimestep_thermal.dt
            time_step_size = microtimestep.dt
            microtimestep_form = microtimestep.after
            microtimestep_form_before = microtimestep
            if m == 0 and macrotimestep_thermal.after is None:
                time_step_size_old = microtimestep.dt
                microtimestep_form_after = microtimestep_form
            elif m == 0:
                time_step_size_old = (
                    macrotimestep_thermal.microtimestep_after.before.dt
                )
                microtimestep_form_after = (
                    macrotimestep_thermal.microtimestep_after
                )
            else:
                time_step_size_old = microtimestep.after.dt
                microtimestep_form_after = microtimestep_form.after

        else:

            extrapolation_proportion = (
                macrotimestep_thermal.tail.point - microtimestep.after.point
            ) / macrotimestep_thermal.dt
            time_step_size = microtimestep.dt
            time_step_size_old = microtimestep.dt
            microtimestep_form_before = None
            microtimestep_form = None
            microtimestep_form_after = None

        # Define intermediate solutions
        velocity_new.assign(
            project(
                extrapolation_proportion * velocity.old
                + (1.0 - extrapolation_proportion) * velocity.new,
                structure.function_space_split[0],
            )
        )
        displacement_new.assign(
            project(
                extrapolation_proportion * displacement.old
                + (1.0 - extrapolation_proportion) * displacement.new,
                structure.function_space_split[1],
            )
        )

        # Define primal solutions for the adjoint problem
        if adjoint:
            velocity_primal = macrotimestep_structure.head.functions[
                "primal_velocity"
            ]
            displacement_primal = macrotimestep_structure.head.functions[
                "primal_displacement"
            ]
            velocity_primal_after = macrotimestep_structure.tail.functions[
                "primal_velocity"
            ]
            displacement_primal_after = macrotimestep_structure.tail.functions[
                "primal_displacement"
            ]
            velocity_primal_new.assign(
                project(
                    extrapolation_proportion * velocity_primal_after
                    + (1.0 - extrapolation_proportion) * velocity_primal,
                    structure.function_space_split[0],
                )
            )
            displacement_primal_new.assign(
                project(
                    extrapolation_proportion * displacement_primal_after
                    + (1.0 - extrapolation_proportion) * displacement_primal,
                    structure.function_space_split[1],
                )
            )
            temperature_primal_new = microtimestep_form.functions[
                "primal_temperature"
            ]
            temperature_primal_old = microtimestep_form_after.functions[
                "primal_temperature"
            ]

        # Define trial and test functions
        temperature_new = TrialFunction(thermal.function_space)
        test_function = TestFunction(thermal.function_space)

        # Define scheme
        time = microtimestep.after.point
        time_before = microtimestep.point
        initial = False
        if not adjoint:
            bilinear_form = thermal.primal_problem.bilinear_form
            functional = thermal.primal_problem.functional
        else:
            bilinear_form = thermal.adjoint_problem.bilinear_form
            functional = thermal.adjoint_problem.functional
            if first_time_step and m == 0:
                initial = True
        if not adjoint:
            left_hand_side = bilinear_form(
                temperature_new,
                test_function,
                structure,
                thermal,
                param,
                time_step_size,
            )
            right_hand_side = functional(
                velocity_new,
                displacement_new,
                temperature_new,
                velocity_old,
                displacement_old,
                temperature_old,
                test_function,
                structure,
                thermal,
                param,
                time_step_size,
            )
        else:
            left_hand_side = bilinear_form(
                velocity_primal_new,
                displacement_primal_new,
                temperature_primal_new,
                temperature_new,
                test_function,
                structure,
                thermal,
                param,
                time_step_size,
                microtimestep_form,
            )
            right_hand_side = functional(
                velocity_primal_new,
                displacement_primal_new,
                temperature_primal_new,
                velocity_primal_old,
                displacement_primal_old,
                temperature_primal_old,
                velocity_new,
                displacement_new,
                temperature_new,
                velocity_old,
                displacement_old,
                temperature_old,
                test_function,
                structure,
                thermal,
                param,
                time_step_size,
                time_step_size_old,
                microtimestep_form_before,
                microtimestep_form,
                microtimestep_form_after,
                primal_solutions,
                local_counter,
                initial,
            )
        right_hand_side_assemble = assemble(right_hand_side)

        # Solve problem
        left_hand_side_assemble = assemble(left_hand_side)
        temperature_new = Function(thermal.function_space)
        solve(
            left_hand_side_assemble,
            temperature_new.vector(),
            right_hand_side_assemble,
        )

        # Save solutions
        if save:

            temperature.save(temperature_new)

        # Update solution
        temperature_old.assign(temperature_new)

        # Update boundary conditions
        velocity_old.assign(velocity_new)
        displacement_old.assign(displacement_new)

        # Advance timeline
        local_counter += 1
        if adjoint:

            microtimestep = microtimestep.before

        else:

            microtimestep = microtimestep.after

        # Update primal solutions for the adjoint problem
        if adjoint:
            velocity_primal_new.assign(velocity_primal_old)
            displacement_primal_new.assign(displacement_primal_old)

    # Save final values
    temperature.new.assign(temperature_new)

    return
