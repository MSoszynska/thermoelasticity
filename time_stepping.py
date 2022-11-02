from fenics import (
    Function,
    UserExpression,
    DirichletBC,
    Constant,
    assemble,
    inner,
    Expression,
    project,
    File,
    dot,
    TestFunction,
    split,
    det,
    inv,
    assign,
    ALE,
)
from solve_structure import solve_structure
from solve_thermal import solve_thermal
from initial import Initial
from spaces import Space
from parameters import Parameters
from time_structure import TimeLine
from reconstructions import copy_list, extrapolate_list, time_array
from forms import zeta, epsilon


def time_stepping(
    structure: Space,
    thermal: Space,
    param: Parameters,
    decoupling,
    structure_timeline: TimeLine,
    thermal_timeline: TimeLine,
    adjoint,
):

    # Initialize function objects
    if adjoint:

        velocity_name = "adjoint_velocity"
        displacement_name = "adjoint_displacement"
        temperature_name = "adjoint_temperature"

    else:

        velocity_name = "primal_velocity"
        displacement_name = "primal_displacement"
        temperature_name = "primal_temperature"

    velocity = Initial(
        "structure", velocity_name, structure.function_space_split[0]
    )
    displacement = Initial(
        "structure", displacement_name, structure.function_space_split[1]
    )
    temperature = Initial(
        "thermal", temperature_name, thermal.function_space_split[0]
    )

    # Save initial values for the primal problem
    if not adjoint and param.INITIAL_TIME == 0.0 and param.PARTIAL_LOAD == 0:

        velocity.save(Function(structure.function_space_split[0]))
        displacement.save(Function(structure.function_space_split[1]))
        temperature.save(Function(thermal.function_space_split[0]))

    # Define time pointers
    if adjoint:

        structure_macrotimestep = structure_timeline.tail
        thermal_macrotimestep = thermal_timeline.tail

    else:

        structure_macrotimestep = structure_timeline.head
        thermal_macrotimestep = thermal_timeline.head

    # Partially load solutions
    if not adjoint and param.PARTIAL_LOAD > 0:
        velocity.save(velocity.load(0), True)
        displacement.save(displacement.load(0), True)
        temperature.save(temperature.load(0), True)
        temp_structure_macrotimestep = structure_macrotimestep
        temp_thermal_macrotimestep = thermal_macrotimestep
        temp_structure_microtimestep = temp_structure_macrotimestep.head.after
        temp_thermal_microtimestep = temp_thermal_macrotimestep.head.after
        structure_counter = 0
        thermal_counter = 0
        left_load = param.PARTIAL_LOAD
        stop_load = False
        while not stop_load:
            if (temp_structure_microtimestep is None) and (
                temp_thermal_microtimestep is None
            ):
                left_load -= 1
                if left_load == 0:
                    stop_load = True
                temp_structure_macrotimestep = (
                    temp_structure_macrotimestep.after
                )
                temp_thermal_macrotimestep = temp_thermal_macrotimestep.after
                temp_structure_microtimestep = (
                    temp_structure_macrotimestep.head.after
                )
                temp_thermal_microtimestep = (
                    temp_thermal_macrotimestep.head.after
                )
            if temp_structure_microtimestep is not None and not stop_load:
                temp_structure_microtimestep = (
                    temp_structure_microtimestep.after
                )
                structure_counter += 1
                velocity.save(velocity.load(structure_counter), True)
                displacement.save(displacement.load(structure_counter), True)
            if temp_thermal_microtimestep is not None and not stop_load:
                temp_thermal_microtimestep = temp_thermal_microtimestep.after
                thermal_counter += 1
                temperature.save(temperature.load(thermal_counter), True)
        structure_macrotimestep = temp_structure_macrotimestep
        thermal_macrotimestep = temp_thermal_macrotimestep
        velocity.old_old = velocity.load(structure_counter)
        displacement.old_old = displacement.load(structure_counter)
        temperature.old_old = temperature.load(thermal_counter)
        velocity.old = velocity.load(structure_counter)
        displacement.old = displacement.load(structure_counter)
        temperature.old = temperature.load(thermal_counter)
        velocity.new = velocity.load(structure_counter)
        displacement.new = displacement.load(structure_counter)
        temperature.new = temperature.load(thermal_counter)
        velocity.HDF5_counter = structure_counter + 1
        displacement.HDF5_counter = structure_counter + 1
        temperature.HDF5_counter = thermal_counter + 1

    if not adjoint and param.INITIAL_TIME > 0.0 and param.PARTIAL_LOAD == 0:
        counter = param.INITIAL_COUNTER
        velocity.old_old = velocity.load(counter)
        displacement.old_old = displacement.load(counter)
        temperature.old_old = temperature.load(counter)
        velocity.old = velocity.load(counter)
        displacement.old = displacement.load(counter)
        temperature.old = temperature.load(counter)
        velocity.new = velocity.load(counter)
        displacement.new = displacement.load(counter)
        temperature.new = temperature.load(counter)
        velocity.save(velocity.new)
        displacement.save(displacement.new)
        temperature.save(temperature.new)

    if adjoint:

        # Collect primal solutions
        velocity_structure_array = copy_list(
            structure_timeline,
            "primal_velocity",
            param,
            False,
        )
        displacement_structure_array = copy_list(
            structure_timeline,
            "primal_displacement",
            param,
            False,
        )
        temperature_structure_array = extrapolate_list(
            structure,
            structure_timeline,
            thermal,
            thermal_timeline,
            "primal_temperature",
            0,
            param,
            False,
        )
        velocity_thermal_array = extrapolate_list(
            thermal,
            thermal_timeline,
            structure,
            structure_timeline,
            "primal_velocity",
            0,
            param,
            False,
        )
        displacement_thermal_array = extrapolate_list(
            thermal,
            thermal_timeline,
            structure,
            structure_timeline,
            "primal_displacement",
            1,
            param,
            False,
        )
        temperature_thermal_array = copy_list(
            thermal_timeline,
            "primal_temperature",
            param,
            False,
        )
    else:
        velocity_structure_array = None
        displacement_structure_array = None
        temperature_structure_array = None
        velocity_thermal_array = None
        displacement_thermal_array = None
        temperature_thermal_array = None

    class Primal_Solutions:
        def __init__(self, structure_timelne, thermal_timeline):
            self.velocity_structure = velocity_structure_array
            self.displacement_structure = displacement_structure_array
            self.temperature_structure = temperature_structure_array
            self.velocity_thermal = velocity_thermal_array
            self.displacement_thermal = displacement_thermal_array
            self.temperature_thermal = temperature_thermal_array
            self.time_structure = time_array(structure_timelne)
            self.time_thermal = time_array(thermal_timeline)

    primal_solutions = Primal_Solutions(structure_timeline, thermal_timeline)

    # Create time loop
    size = structure_timeline.size - param.PARTIAL_LOAD
    if param.PARTIAL_LOAD == 0:
        first_time_step = True
    else:
        first_time_step = False
    counter = param.PARTIAL_LOAD
    counter_structure = counter
    counter_thermal = counter
    time_test = 0
    for n in range(size):

        if adjoint:

            print(f"Current macro time-step {size - counter}")

        else:

            print(f"Current macro time-step {counter + 1}")

        # Perform decoupling
        decoupling(
            velocity,
            displacement,
            temperature,
            first_time_step,
            structure,
            thermal,
            param,
            structure_macrotimestep,
            thermal_macrotimestep,
            adjoint,
            counter_structure,
            counter_thermal,
            primal_solutions,
        )

        # Perform final iteration and save solutions
        solve_structure(
            velocity,
            displacement,
            temperature,
            structure,
            thermal,
            first_time_step,
            param,
            structure_macrotimestep,
            thermal_macrotimestep,
            adjoint,
            counter_structure,
            primal_solutions,
            save=True,
        )
        solve_thermal(
            velocity,
            displacement,
            temperature,
            structure,
            thermal,
            first_time_step,
            param,
            structure_macrotimestep,
            thermal_macrotimestep,
            adjoint,
            counter_thermal,
            primal_solutions,
            save=True,
        )
        first_time_step = False

        functional_test = (
            inner(
                epsilon(displacement.new),
                epsilon(displacement.new),
            )
            * structure.dx
        )
        time_test += 0.002
        print(f"(" f"{time_test}" f"," f"{assemble(functional_test)}" f")")

        # Update solution
        velocity.old_old.assign(velocity.old)
        displacement.old_old.assign(displacement.old)
        temperature.old_old.assign(temperature.old)
        velocity.old.assign(velocity.new)
        displacement.old.assign(displacement.new)
        temperature.old.assign(temperature.new)

        # Advance timeline
        if adjoint:

            structure_macrotimestep = structure_macrotimestep.before
            thermal_macrotimestep = thermal_macrotimestep.before

        else:

            structure_macrotimestep = structure_macrotimestep.after
            thermal_macrotimestep = thermal_macrotimestep.after
        counter += 1
        if structure_macrotimestep is not None:
            counter_structure += structure_macrotimestep.size - 1
        if thermal_macrotimestep is not None:
            counter_thermal += thermal_macrotimestep.size - 1

    # Save initial values for the adjoint problem
    if adjoint:

        velocity.save(Function(structure.function_space_split[0]))
        displacement.save(Function(structure.function_space_split[1]))
        temperature.save(Function(thermal.function_space_split[0]))

    # Check convergence
    [print(linear_systems) for linear_systems in velocity.iterations]
    failed = 0
    for i in range(len(velocity.iterations)):
        failed += min(0, velocity.iterations[i])
    if failed < 0:
        print("The decoupling method failed at some point")
