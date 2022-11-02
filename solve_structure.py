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
    det,
    inv,
    dot,
)
from spaces import Space
from parameters import Parameters
from time_structure import MacroTimeStep
from initial import Initial


def solve_structure(
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

    # Store old structure solutions
    velocity_old = Function(structure.function_space_split[0])
    displacement_old = Function(structure.function_space_split[1])
    velocity_old.assign(velocity.old)
    displacement_old.assign(displacement.old)

    # Store old thermal solutions
    temperature_old = Function(thermal.function_space_split[0])
    temperature_old.assign(temperature.old)

    # Initialize new thermal solutions
    temperature_new = Function(thermal.function_space_split[0])

    # Initialize primal solutions for the adjoint problem
    if adjoint:
        temperature_primal_new = Function(thermal.function_space_split[0])
        temperature_primal_old = Function(thermal.function_space_split[0])
        temperature_primal_old.assign(
            macrotimestep_thermal.tail.functions["primal_temperature"]
        )

    # Define time pointers
    if adjoint:

        microtimestep = macrotimestep_structure.tail.before

    else:

        microtimestep = macrotimestep_structure.head

    # Compute macro time-step size
    size = macrotimestep_structure.size - 1
    local_counter = counter
    for m in range(size):

        # Extrapolate weak boundary conditions on the interface
        if adjoint:

            extrapolation_proportion = (
                microtimestep.point - macrotimestep_structure.head.point
            ) / macrotimestep_structure.dt
            time_step_size = microtimestep.dt
            microtimestep_form = microtimestep.after
            microtimestep_form_before = microtimestep
            if m == 0 and macrotimestep_structure.after is None:
                time_step_size_old = microtimestep.dt
                microtimestep_form_after = microtimestep_form
            elif m == 0:
                time_step_size_old = (
                    macrotimestep_structure.microtimestep_after.before.dt
                )
                microtimestep_form_after = (
                    macrotimestep_structure.microtimestep_after
                )
            else:
                time_step_size_old = microtimestep.after.dt
                microtimestep_form_after = microtimestep_form.after

        else:

            extrapolation_proportion = (
                macrotimestep_structure.tail.point - microtimestep.after.point
            ) / macrotimestep_structure.dt
            time_step_size = microtimestep.dt
            time_step_size_old = microtimestep.dt
            microtimestep_form_before = None
            microtimestep_form = None
            microtimestep_form_after = None

        # Define intermediate solutions
        temperature_new.assign(
            project(
                extrapolation_proportion * temperature.old
                + (1.0 - extrapolation_proportion) * temperature.new,
                thermal.function_space_split[0],
            )
        )

        if adjoint:
            temperature_primal = macrotimestep_thermal.head.functions[
                "primal_temperature"
            ]
            temperature_primal_after = macrotimestep_thermal.tail.functions[
                "primal_temperature"
            ]
            temperature_primal_new.assign(
                project(
                    extrapolation_proportion * temperature_primal_after
                    + (1.0 - extrapolation_proportion) * temperature_primal,
                    thermal.function_space_split[0],
                )
            )
            velocity_primal_new = microtimestep_form.functions[
                "primal_velocity"
            ]
            velocity_primal_old = microtimestep_form_after.functions[
                "primal_velocity"
            ]
            displacement_primal_new = microtimestep_form.functions[
                "primal_displacement"
            ]
            displacement_primal_old = microtimestep_form_after.functions[
                "primal_displacement"
            ]

        # Define trial and test functions
        trial_function = TrialFunction(structure.function_space)
        (
            velocity_new,
            displacement_new,
        ) = split(trial_function)
        test_function = TestFunction(structure.function_space)
        (
            first_test_function,
            second_test_function,
        ) = split(test_function)

        # Define scheme
        time = microtimestep.after.point
        time_before = microtimestep.point
        initial = False
        if not adjoint:
            bilinear_form = structure.primal_problem.bilinear_form
            functional = structure.primal_problem.functional
        else:
            bilinear_form = structure.adjoint_problem.bilinear_form
            functional = structure.adjoint_problem.functional
            if first_time_step and m == 0:
                initial = True
        if not adjoint:
            left_hand_side = bilinear_form(
                velocity_new,
                displacement_new,
                temperature_new,
                first_test_function,
                second_test_function,
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
                first_test_function,
                second_test_function,
                structure,
                thermal,
                param,
                time,
                time_before,
                time_step_size,
            )
        else:
            left_hand_side = bilinear_form(
                velocity_primal_new,
                displacement_primal_new,
                temperature_primal_new,
                velocity_new,
                displacement_new,
                temperature_new,
                first_test_function,
                second_test_function,
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
                first_test_function,
                second_test_function,
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
        time = microtimestep.after.point
        left_hand_side_assemble = assemble(left_hand_side)
        trial_function = Function(structure.function_space)
        [
            boundary.apply(left_hand_side_assemble, right_hand_side_assemble)
            for boundary in structure.boundaries()
        ]
        solve(
            left_hand_side_assemble,
            trial_function.vector(),
            right_hand_side_assemble,
        )
        (
            velocity_new,
            displacement_new,
        ) = trial_function.split(trial_function)

        # Save solutions
        if save:
            velocity.save(velocity_new)
            displacement.save(displacement_new)

        # Update structure solutions
        velocity_old.assign(velocity_new)
        displacement_old.assign(displacement_new)

        # Update thermal solutions
        temperature_old.assign(temperature_new)

        # Advance timeline
        local_counter += 1
        if adjoint:

            microtimestep = microtimestep.before

        else:

            microtimestep = microtimestep.after

        # Update primal solutions for the adjoint problem
        if adjoint:
            temperature_primal_new.assign(temperature_primal_old)

    # Save final values
    velocity.new.assign(velocity_new)
    displacement.new.assign(displacement_new)

    return
