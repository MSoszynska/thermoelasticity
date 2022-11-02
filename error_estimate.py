from fenics import (
    project,
    assemble,
    dot,
    grad,
    inner,
    Function,
    assign,
    Constant,
    File,
    near,
    DirichletBC,
    Constant,
    Point,
    action,
    div,
)
from forms import (
    form_structure,
    form_thermal,
    form_structure_adjoint,
    form_thermal_adjoint,
    form_structure_thermal_adjoint,
    external_force,
    epsilon,
    zeta,
    zeta_adjoint,
)
from parameters import Parameters
from spaces import Space
from time_structure import MicroTimeStep, MacroTimeStep, TimeLine
from reconstructions import (
    gauss,
    copy_list,
    extrapolate_list,
    linear_extrapolation,
    primal_reconstruction,
    primal_derivative,
    adjoint_reconstruction,
)


# Compute goal functionals
def goal_functional_structure(
    structure: Space,
    structure_timeline: TimeLine,
    thermal: Space,
    thermal_timeline: TimeLine,
    param: Parameters,
):

    if param.PARTIAL_COMPUTE is not None:
        starting_point = param.PARTIAL_COMPUTE[0]
        size = param.PARTIAL_COMPUTE[1]
        macrotimestep_adjust = structure_timeline.head
        for i in range(starting_point):
            macrotimestep_adjust = macrotimestep_adjust.after
        macrotimestep = macrotimestep_adjust
        global_size = size
    else:
        macrotimestep = structure_timeline.head
        global_size = structure_timeline.size

    # Prepare arrays of solutions
    displacement_array = copy_list(
        structure_timeline, "primal_displacement", param, False
    )
    temperature_array = extrapolate_list(
        structure,
        structure_timeline,
        thermal,
        thermal_timeline,
        "primal_temperature",
        0,
        param,
        False,
    )

    goal_functional = []
    m = 1
    for n in range(global_size):

        microtimestep = macrotimestep.head.after
        local_size = macrotimestep.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            result = 0.0
            time_step_size = microtimestep.before.dt
            gauss_1, gauss_2 = gauss(microtimestep)
            result += (
                0.5
                * time_step_size
                * zeta(
                    linear_extrapolation(
                        temperature_array, m, gauss_1, microtimestep
                    ),
                    param,
                )
                * param.GOAL_FUNCTIONAL_STRUCTURE
                * inner(
                    epsilon(
                        linear_extrapolation(
                            displacement_array, m, gauss_1, microtimestep
                        )
                    ),
                    epsilon(
                        linear_extrapolation(
                            displacement_array, m, gauss_1, microtimestep
                        )
                    ),
                )
                * structure.dx
            )
            result += (
                0.5
                * time_step_size
                * zeta(
                    linear_extrapolation(
                        temperature_array, m, gauss_2, microtimestep
                    ),
                    param,
                )
                * param.GOAL_FUNCTIONAL_STRUCTURE
                * inner(
                    epsilon(
                        linear_extrapolation(
                            displacement_array, m, gauss_2, microtimestep
                        )
                    ),
                    epsilon(
                        linear_extrapolation(
                            displacement_array, m, gauss_2, microtimestep
                        )
                    ),
                )
                * structure.dx
            )
            goal_functional.append(assemble(result))
            m += 1
            microtimestep = microtimestep.after

        macrotimestep = macrotimestep.after

    return goal_functional


def goal_functional_thermal(
    thermal,
    thermal_timeline,
    param,
):

    if param.PARTIAL_COMPUTE is not None:
        starting_point = param.PARTIAL_COMPUTE[0]
        size = param.PARTIAL_COMPUTE[1]
        macrotimestep_adjust = thermal_timeline.head
        for i in range(starting_point):
            macrotimestep_adjust = macrotimestep_adjust.after
        macrotimestep = macrotimestep_adjust
        global_size = size
    else:
        macrotimestep = thermal_timeline.head
        global_size = thermal_timeline.size

    # Prepare arrays of solutions
    temperature_array = copy_list(
        thermal_timeline, "primal_temperature", param, False
    )
    goal_functional = []
    m = 1
    for n in range(global_size):

        microtimestep = macrotimestep.head.after
        local_size = macrotimestep.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            result = 0.0
            time_step_size = microtimestep.before.dt
            gauss_1, gauss_2 = gauss(microtimestep)
            result += (
                0.5
                * time_step_size
                * param.KAPPA
                * param.GOAL_FUNCTIONAL_THERMAL
                * dot(
                    linear_extrapolation(
                        temperature_array, m, gauss_1, microtimestep
                    ),
                    linear_extrapolation(
                        temperature_array, m, gauss_1, microtimestep
                    ),
                )
                * thermal.dx
            )
            result += (
                0.5
                * time_step_size
                * param.KAPPA
                * param.GOAL_FUNCTIONAL_THERMAL
                * dot(
                    linear_extrapolation(
                        temperature_array, m, gauss_2, microtimestep
                    ),
                    linear_extrapolation(
                        temperature_array, m, gauss_2, microtimestep
                    ),
                )
                * thermal.dx
            )
            goal_functional.append(assemble(result))
            m += 1
            microtimestep = microtimestep.after

        macrotimestep = macrotimestep.after

    return goal_functional


# Compute primal residual of the structure subproblem
def primal_residual_structure(
    structure: Space,
    thermal: Space,
    structure_timeline: TimeLine,
    thermal_timeline: TimeLine,
    param: Parameters,
):

    if param.PARTIAL_COMPUTE is not None:
        if param.PARTIAL_COMPUTE[0] != 0:
            starting_point = param.PARTIAL_COMPUTE[0] - 2
            size = param.PARTIAL_COMPUTE[1] + 2
        else:
            starting_point = param.PARTIAL_COMPUTE[0]
            size = param.PARTIAL_COMPUTE[1]
        macrotimestep_structure_adjust = structure_timeline.head
        for i in range(starting_point):
            macrotimestep_structure_adjust = (
                macrotimestep_structure_adjust.after
            )
        macrotimestep_structure = macrotimestep_structure_adjust
        global_size = size
    else:
        macrotimestep_structure = structure_timeline.head
        global_size = structure_timeline.size

    # Prepare arrays of solutions
    velocity_array = copy_list(
        structure_timeline, "primal_velocity", param, True
    )
    displacement_array = copy_list(
        structure_timeline, "primal_displacement", param, True
    )
    temperature_array = extrapolate_list(
        structure,
        structure_timeline,
        thermal,
        thermal_timeline,
        "primal_temperature",
        0,
        param,
        False,
    )
    velocity_adjoint_array = copy_list(
        structure_timeline,
        "adjoint_velocity",
        param,
        True,
    )
    displacement_adjoint_array = copy_list(
        structure_timeline,
        "adjoint_displacement",
        param,
        True,
    )
    temperature_adjoint_array = extrapolate_list(
        structure,
        structure_timeline,
        thermal,
        thermal_timeline,
        "adjoint_temperature",
        0,
        param,
        True,
    )
    residuals = []
    m = 1
    for n in range(global_size):

        microtimestep_structure_before = macrotimestep_structure.head
        microtimestep_structure = macrotimestep_structure.head.after
        if microtimestep_structure.after is None:
            if macrotimestep_structure.after is not None:
                microtimestep_structure_after = (
                    macrotimestep_structure.after.head.after
                )
            else:
                microtimestep_structure_after = None
        else:
            microtimestep_structure_after = microtimestep_structure.after
        time = microtimestep_structure.point
        local_size = macrotimestep_structure.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            lhs = 0.0
            rhs = 0.0
            time_step_size = microtimestep_structure.before.dt
            gauss_1, gauss_2 = gauss(microtimestep_structure)

            lhs += (
                0.5
                * param.RHO
                * time_step_size
                * dot(
                    (velocity_array[m] - velocity_array[m - 1])
                    / time_step_size,
                    adjoint_reconstruction(
                        velocity_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_structure,
                        macrotimestep_structure,
                    )
                    - velocity_adjoint_array[m],
                )
                * structure.dx
            )
            lhs += (
                0.5
                * param.RHO
                * time_step_size
                * dot(
                    (velocity_array[m] - velocity_array[m - 1])
                    / time_step_size,
                    adjoint_reconstruction(
                        velocity_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_structure,
                        macrotimestep_structure,
                    )
                    - velocity_adjoint_array[m],
                )
                * structure.dx
            )
            lhs += (
                0.5
                * time_step_size
                * dot(
                    (displacement_array[m] - displacement_array[m - 1])
                    / time_step_size,
                    adjoint_reconstruction(
                        displacement_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_structure,
                        macrotimestep_structure,
                    )
                    - displacement_adjoint_array[m],
                )
                * structure.dx
            )
            lhs += (
                0.5
                * time_step_size
                * dot(
                    (displacement_array[m] - displacement_array[m - 1])
                    / time_step_size,
                    adjoint_reconstruction(
                        displacement_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_structure,
                        macrotimestep_structure,
                    )
                    - displacement_adjoint_array[m],
                )
                * structure.dx
            )
            lhs += (
                0.5
                * time_step_size
                * form_structure(
                    linear_extrapolation(
                        velocity_array, m, gauss_1, microtimestep_structure
                    ),
                    linear_extrapolation(
                        displacement_array,
                        m,
                        gauss_1,
                        microtimestep_structure,
                    ),
                    linear_extrapolation(
                        temperature_array,
                        m,
                        gauss_1,
                        microtimestep_structure,
                    ),
                    adjoint_reconstruction(
                        velocity_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_structure,
                        macrotimestep_structure,
                    )
                    - velocity_adjoint_array[m],
                    adjoint_reconstruction(
                        displacement_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_structure,
                        macrotimestep_structure,
                    )
                    - displacement_adjoint_array[m],
                    structure,
                    param,
                )
            )
            lhs += (
                0.5
                * time_step_size
                * form_structure(
                    linear_extrapolation(
                        velocity_array, m, gauss_2, microtimestep_structure
                    ),
                    linear_extrapolation(
                        displacement_array,
                        m,
                        gauss_2,
                        microtimestep_structure,
                    ),
                    linear_extrapolation(
                        temperature_array,
                        m,
                        gauss_2,
                        microtimestep_structure,
                    ),
                    adjoint_reconstruction(
                        velocity_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_structure,
                        macrotimestep_structure,
                    )
                    - velocity_adjoint_array[m],
                    adjoint_reconstruction(
                        displacement_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_structure,
                        macrotimestep_structure,
                    )
                    - displacement_adjoint_array[m],
                    structure,
                    param,
                )
            )
            rhs += (
                0.5
                * param.RHO
                * time_step_size
                * dot(
                    external_force(gauss_1),
                    (
                        adjoint_reconstruction(
                            velocity_adjoint_array,
                            m,
                            gauss_1,
                            microtimestep_structure,
                            macrotimestep_structure,
                        )
                        - velocity_adjoint_array[m]
                    ),
                )
                * structure.dx
            )
            rhs += (
                0.5
                * param.RHO
                * time_step_size
                * dot(
                    external_force(gauss_2),
                    (
                        adjoint_reconstruction(
                            velocity_adjoint_array,
                            m,
                            gauss_2,
                            microtimestep_structure,
                            macrotimestep_structure,
                        )
                        - velocity_adjoint_array[m]
                    ),
                )
                * structure.dx
            )
            residuals.append(assemble(0.5 * rhs - 0.5 * lhs))
            m += 1
            microtimestep_structure_before = (
                microtimestep_structure_before.after
            )
            microtimestep_structure = microtimestep_structure.after

        macrotimestep_structure = macrotimestep_structure.after

    return residuals


# Compute primal residual of the thermal subproblem
def primal_residual_thermal(
    thermal: Space,
    structure: Space,
    thermal_timeline: TimeLine,
    structure_timeline: TimeLine,
    param: Parameters,
):

    if param.PARTIAL_COMPUTE is not None:
        if param.PARTIAL_COMPUTE[0] != 0:
            starting_point = param.PARTIAL_COMPUTE[0] - 2
            size = param.PARTIAL_COMPUTE[1] + 2
        else:
            starting_point = param.PARTIAL_COMPUTE[0]
            size = param.PARTIAL_COMPUTE[1]
        macrotimestep_thermal_adjust = thermal_timeline.head
        for i in range(starting_point):
            macrotimestep_thermal_adjust = macrotimestep_thermal_adjust.after
        macrotimestep_thermal = macrotimestep_thermal_adjust
        global_size = size
    else:
        macrotimestep_thermal = thermal_timeline.head
        global_size = thermal_timeline.size

    # Prepare arrays of solutions
    temperature_array = copy_list(
        thermal_timeline, "primal_temperature", param, True
    )
    displacement_array = extrapolate_list(
        thermal,
        thermal_timeline,
        structure,
        structure_timeline,
        "primal_displacement",
        1,
        param,
        True,
    )
    temperature_adjoint_array = copy_list(
        thermal_timeline,
        "adjoint_temperature",
        param,
        True,
    )
    residuals = []
    m = 1
    for n in range(global_size):

        microtimestep_thermal_before = macrotimestep_thermal.head
        microtimestep_thermal = macrotimestep_thermal.head.after
        if microtimestep_thermal.after is None:
            if macrotimestep_thermal.after is not None:
                microtimestep_thermal_after = (
                    macrotimestep_thermal.after.head.after
                )
            else:
                microtimestep_thermal_after = None
        else:
            microtimestep_thermal_after = microtimestep_thermal.after
        time = microtimestep_thermal.point
        local_size = macrotimestep_thermal.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            lhs = 0.0
            rhs = 0.0
            time_step_size = microtimestep_thermal.before.dt
            gauss_1, gauss_2 = gauss(microtimestep_thermal)

            lhs += (
                0.5
                * param.RHO
                * param.C
                * time_step_size
                * dot(
                    (temperature_array[m] - temperature_array[m - 1])
                    / time_step_size,
                    adjoint_reconstruction(
                        temperature_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_thermal,
                        macrotimestep_thermal,
                    )
                    - temperature_adjoint_array[m],
                )
                * thermal.dx
            )
            lhs += (
                0.5
                * param.RHO
                * param.C
                * time_step_size
                * dot(
                    (temperature_array[m] - temperature_array[m - 1])
                    / time_step_size,
                    adjoint_reconstruction(
                        temperature_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_thermal,
                        macrotimestep_thermal,
                    )
                    - temperature_adjoint_array[m],
                )
                * thermal.dx
            )
            lhs += (
                0.5
                * time_step_size
                * form_thermal(
                    linear_extrapolation(
                        temperature_array, m, gauss_1, microtimestep_thermal
                    ),
                    adjoint_reconstruction(
                        temperature_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_thermal,
                        macrotimestep_thermal,
                    )
                    - temperature_adjoint_array[m],
                    structure,
                    thermal,
                    param,
                )
            )
            lhs += (
                0.5
                * time_step_size
                * form_thermal(
                    linear_extrapolation(
                        temperature_array, m, gauss_2, microtimestep_thermal
                    ),
                    adjoint_reconstruction(
                        temperature_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_thermal,
                        macrotimestep_thermal,
                    )
                    - temperature_adjoint_array[m],
                    structure,
                    thermal,
                    param,
                )
            )
            rhs += (
                0.5
                * time_step_size
                * param.ALPHA
                * dot(
                    div(
                        linear_extrapolation(
                            displacement_array,
                            m,
                            gauss_1,
                            microtimestep_thermal,
                        )
                    ),
                    adjoint_reconstruction(
                        temperature_adjoint_array,
                        m,
                        gauss_1,
                        microtimestep_thermal,
                        macrotimestep_thermal,
                    )
                    - temperature_adjoint_array[m],
                )
                * thermal.dx
            )
            rhs += (
                0.5
                * time_step_size
                * param.ALPHA
                * dot(
                    div(
                        linear_extrapolation(
                            displacement_array,
                            m,
                            gauss_2,
                            microtimestep_thermal,
                        )
                    ),
                    adjoint_reconstruction(
                        temperature_adjoint_array,
                        m,
                        gauss_2,
                        microtimestep_thermal,
                        macrotimestep_thermal,
                    )
                    - temperature_adjoint_array[m],
                )
                * thermal.dx
            )

            residuals.append(assemble(0.5 * rhs - 0.5 * lhs))
            m += 1
            microtimestep_thermal_before = microtimestep_thermal_before.after
            microtimestep_thermal = microtimestep_thermal.after

        macrotimestep_thermal = macrotimestep_thermal.after

    return residuals


# Compute adjoint residual of the structure subproblem
def adjoint_residual_structure(
    structure: Space,
    thermal: Space,
    structure_timeline: TimeLine,
    thermal_timeline: TimeLine,
    param: Parameters,
):
    if param.PARTIAL_COMPUTE is not None:
        starting_point = param.PARTIAL_COMPUTE[0]
        size = param.PARTIAL_COMPUTE[1]
        macrotimestep_structure_adjust = structure_timeline.head
        for i in range(starting_point):
            macrotimestep_structure_adjust = (
                macrotimestep_structure_adjust.after
            )
        macrotimestep_structure = macrotimestep_structure_adjust
        global_size = size
    else:
        macrotimestep_structure = structure_timeline.head
        global_size = structure_timeline.size

    # Prepare arrays of solutions
    velocity_array = copy_list(
        structure_timeline,
        "primal_velocity",
        param,
        False,
    )
    displacement_array = copy_list(
        structure_timeline,
        "primal_displacement",
        param,
        False,
    )
    velocity_adjoint_array = copy_list(
        structure_timeline, "adjoint_velocity", param, False
    )
    displacement_adjoint_array = copy_list(
        structure_timeline, "adjoint_displacement", param, False
    )
    temperature_array = extrapolate_list(
        structure,
        structure_timeline,
        thermal,
        thermal_timeline,
        "primal_temperature",
        0,
        param,
        False,
    )
    temperature_adjoint_array = extrapolate_list(
        structure,
        structure_timeline,
        thermal,
        thermal_timeline,
        "adjoint_temperature",
        0,
        param,
        True,
    )
    residuals = []
    left = True
    m = 1
    if param.PARTIAL_COMPUTE is not None:
        if param.PARTIAL_COMPUTE[0] != 0:
            residuals.append(0.0)
    for n in range(global_size):

        microtimestep_structure_before = macrotimestep_structure.head
        microtimestep_structure = macrotimestep_structure.head.after
        local_size = macrotimestep_structure.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            lhs = 0.0
            rhs = 0.0
            time_step_size = microtimestep_structure.before.dt
            gauss_1, gauss_2 = gauss(microtimestep_structure)
            if left:

                l = m
                microtimestep_structure_adjust = microtimestep_structure
                left = False

            else:

                l = m - 1
                if microtimestep_structure.before.before is None:

                    microtimestep_structure_adjust = (
                        macrotimestep_structure.microtimestep_before.after
                    )

                else:

                    microtimestep_structure_adjust = (
                        microtimestep_structure.before
                    )
                left = True

            zzeta = (
                (param.NU * param.E_INITIAL)
                / (1.0 + param.NU)
                * (1.0 - 2.0 * param.NU)
            )

            lhs += (
                0.5
                * param.RHO
                * time_step_size
                * dot(
                    primal_derivative(
                        velocity_array,
                        l,
                        gauss_1,
                        microtimestep_structure_adjust,
                    )
                    - (velocity_array[m] - velocity_array[m - 1])
                    / time_step_size,
                    velocity_adjoint_array[m],
                )
                * structure.dx
            )
            lhs += (
                0.5
                * param.RHO
                * time_step_size
                * dot(
                    primal_derivative(
                        velocity_array,
                        l,
                        gauss_2,
                        microtimestep_structure_adjust,
                    )
                    - (velocity_array[m] - velocity_array[m - 1])
                    / time_step_size,
                    velocity_adjoint_array[m],
                )
                * structure.dx
            )
            lhs += (
                0.5
                * time_step_size
                * dot(
                    primal_derivative(
                        displacement_array,
                        l,
                        gauss_1,
                        microtimestep_structure_adjust,
                    )
                    - (displacement_array[m] - displacement_array[m - 1])
                    / time_step_size,
                    displacement_adjoint_array[m],
                )
                * structure.dx
            )
            lhs += (
                0.5
                * time_step_size
                * dot(
                    primal_derivative(
                        displacement_array,
                        l,
                        gauss_2,
                        microtimestep_structure_adjust,
                    )
                    - (displacement_array[m] - displacement_array[m - 1])
                    / time_step_size,
                    displacement_adjoint_array[m],
                )
                * structure.dx
            )
            lhs += (
                0.5
                * time_step_size
                * form_structure_adjoint(
                    linear_extrapolation(
                        velocity_array, m, gauss_1, microtimestep_structure
                    ),
                    linear_extrapolation(
                        displacement_array,
                        m,
                        gauss_1,
                        microtimestep_structure,
                    ),
                    linear_extrapolation(
                        temperature_array,
                        m,
                        gauss_1,
                        microtimestep_structure,
                    ),
                    velocity_adjoint_array[m],
                    displacement_adjoint_array[m],
                    primal_reconstruction(
                        velocity_array,
                        l,
                        gauss_1,
                        microtimestep_structure_adjust,
                    )
                    - linear_extrapolation(
                        velocity_array, m, gauss_1, microtimestep_structure
                    ),
                    primal_reconstruction(
                        displacement_array,
                        l,
                        gauss_1,
                        microtimestep_structure_adjust,
                    )
                    - linear_extrapolation(
                        displacement_array,
                        m,
                        gauss_1,
                        microtimestep_structure,
                    ),
                    structure,
                    thermal,
                    param,
                )
            )
            lhs += (
                0.5
                * time_step_size
                * form_structure_adjoint(
                    linear_extrapolation(
                        velocity_array, m, gauss_2, microtimestep_structure
                    ),
                    linear_extrapolation(
                        displacement_array,
                        m,
                        gauss_2,
                        microtimestep_structure,
                    ),
                    linear_extrapolation(
                        temperature_array,
                        m,
                        gauss_2,
                        microtimestep_structure,
                    ),
                    velocity_adjoint_array[m],
                    displacement_adjoint_array[m],
                    primal_reconstruction(
                        velocity_array,
                        l,
                        gauss_2,
                        microtimestep_structure_adjust,
                    )
                    - linear_extrapolation(
                        velocity_array, m, gauss_2, microtimestep_structure
                    ),
                    primal_reconstruction(
                        displacement_array,
                        l,
                        gauss_2,
                        microtimestep_structure_adjust,
                    )
                    - linear_extrapolation(
                        displacement_array,
                        m,
                        gauss_2,
                        microtimestep_structure,
                    ),
                    structure,
                    thermal,
                    param,
                )
            )
            rhs += (
                0.5
                * time_step_size
                * 2.0
                * zeta(
                    linear_extrapolation(
                        temperature_array, m, gauss_1, microtimestep_structure
                    ),
                    param,
                )
                * inner(
                    epsilon(
                        linear_extrapolation(
                            displacement_array,
                            m,
                            gauss_1,
                            microtimestep_structure,
                        )
                    ),
                    epsilon(
                        primal_reconstruction(
                            displacement_array,
                            l,
                            gauss_1,
                            microtimestep_structure_adjust,
                        )
                        - linear_extrapolation(
                            displacement_array,
                            m,
                            gauss_1,
                            microtimestep_structure,
                        )
                    ),
                )
                * structure.dx
            )
            rhs += (
                0.5
                * time_step_size
                * 2.0
                * zeta(
                    linear_extrapolation(
                        temperature_array, m, gauss_2, microtimestep_structure
                    ),
                    param,
                )
                * inner(
                    epsilon(
                        linear_extrapolation(
                            displacement_array,
                            m,
                            gauss_2,
                            microtimestep_structure,
                        )
                    ),
                    epsilon(
                        primal_reconstruction(
                            displacement_array,
                            l,
                            gauss_2,
                            microtimestep_structure_adjust,
                        )
                        - linear_extrapolation(
                            displacement_array,
                            m,
                            gauss_2,
                            microtimestep_structure,
                        )
                    ),
                )
                * structure.dx
            )
            rhs += (
                0.5
                * time_step_size
                * param.ALPHA
                * dot(
                    temperature_adjoint_array[m],
                    div(
                        primal_reconstruction(
                            displacement_array,
                            l,
                            gauss_1,
                            microtimestep_structure_adjust,
                        )
                        - linear_extrapolation(
                            displacement_array,
                            m,
                            gauss_1,
                            microtimestep_structure,
                        )
                    ),
                )
                * structure.dx
            )
            rhs += (
                0.5
                * time_step_size
                * param.ALPHA
                * dot(
                    temperature_adjoint_array[m],
                    div(
                        primal_reconstruction(
                            displacement_array,
                            l,
                            gauss_2,
                            microtimestep_structure_adjust,
                        )
                        - linear_extrapolation(
                            displacement_array,
                            m,
                            gauss_2,
                            microtimestep_structure,
                        )
                    ),
                )
                * structure.dx
            )

            residuals.append(assemble(0.5 * rhs - 0.5 * lhs))
            m += 1
            microtimestep_structure_before = (
                microtimestep_structure_before.after
            )
            microtimestep_structure = microtimestep_structure.after

        macrotimestep_structure = macrotimestep_structure.after
    if param.PARTIAL_COMPUTE is not None:
        if param.PARTIAL_COMPUTE[0] != 0:
            residuals.append(0.0)

    return residuals


# Compute adjoint residual of the thermal subproblem
def adjoint_residual_thermal(
    thermal: Space,
    structure: Space,
    thermal_timeline: TimeLine,
    structure_timeline: TimeLine,
    param: Parameters,
):

    if param.PARTIAL_COMPUTE is not None:
        starting_point = param.PARTIAL_COMPUTE[0]
        size = param.PARTIAL_COMPUTE[1]
        macrotimestep_thermal_adjust = thermal_timeline.head
        for i in range(starting_point):
            macrotimestep_thermal_adjust = macrotimestep_thermal_adjust.after
        macrotimestep_thermal = macrotimestep_thermal_adjust
        global_size = size
    else:
        macrotimestep_thermal = thermal_timeline.head
        global_size = thermal_timeline.size

    # Prepare arrays of solutions
    velocity_array = extrapolate_list(
        thermal,
        thermal_timeline,
        structure,
        structure_timeline,
        "primal_velocity",
        0,
        param,
        False,
    )
    displacement_array = extrapolate_list(
        thermal,
        thermal_timeline,
        structure,
        structure_timeline,
        "primal_displacement",
        1,
        param,
        False,
    )
    temperature_array = copy_list(
        thermal_timeline,
        "primal_temperature",
        param,
        False,
    )
    velocity_adjoint_array = extrapolate_list(
        thermal,
        thermal_timeline,
        structure,
        structure_timeline,
        "adjoint_velocity",
        0,
        param,
        False,
    )
    displacement_adjoint_array = extrapolate_list(
        thermal,
        thermal_timeline,
        structure,
        structure_timeline,
        "adjoint_displacement",
        1,
        param,
        False,
    )
    temperature_adjoint_array = copy_list(
        thermal_timeline, "adjoint_temperature", param, False
    )
    residuals = []
    left = True
    m = 1
    if param.PARTIAL_COMPUTE is not None:
        if param.PARTIAL_COMPUTE[0] != 0:
            residuals.append(0.0)
    for n in range(global_size):

        microtimestep_thermal_before = macrotimestep_thermal.head
        microtimestep_thermal = macrotimestep_thermal.head.after
        local_size = macrotimestep_thermal.size - 1
        for k in range(local_size):

            print(f"Current contribution: {m}")
            lhs = 0.0
            rhs = 0.0
            time_step_size = microtimestep_thermal.before.dt
            gauss_1, gauss_2 = gauss(microtimestep_thermal)
            if left:

                l = m
                microtimestep_thermal_adjust = microtimestep_thermal
                left = False

            else:

                l = m - 1
                if microtimestep_thermal.before.before is None:

                    microtimestep_thermal_adjust = (
                        macrotimestep_thermal.microtimestep_before.after
                    )

                else:

                    microtimestep_thermal_adjust = microtimestep_thermal.before
                left = True

            lhs += (
                0.5
                * param.C
                * param.RHO
                * time_step_size
                * dot(
                    primal_derivative(
                        temperature_array,
                        l,
                        gauss_1,
                        microtimestep_thermal_adjust,
                    )
                    - (temperature_array[m] - temperature_array[m - 1])
                    / time_step_size,
                    temperature_adjoint_array[m],
                )
                * thermal.dx
            )
            lhs += (
                0.5
                * param.C
                * param.RHO
                * time_step_size
                * dot(
                    primal_derivative(
                        temperature_array,
                        l,
                        gauss_2,
                        microtimestep_thermal_adjust,
                    )
                    - (temperature_array[m] - temperature_array[m - 1])
                    / time_step_size,
                    temperature_adjoint_array[m],
                )
                * thermal.dx
            )
            lhs += (
                0.5
                * time_step_size
                * form_thermal_adjoint(
                    linear_extrapolation(
                        temperature_array, m, gauss_1, microtimestep_thermal
                    ),
                    temperature_adjoint_array[m],
                    primal_reconstruction(
                        temperature_array,
                        l,
                        gauss_1,
                        microtimestep_thermal_adjust,
                    )
                    - linear_extrapolation(
                        temperature_array, m, gauss_1, microtimestep_thermal
                    ),
                    structure,
                    thermal,
                    param,
                )
            )
            lhs += (
                0.5
                * time_step_size
                * form_thermal_adjoint(
                    linear_extrapolation(
                        temperature_array, m, gauss_2, microtimestep_thermal
                    ),
                    temperature_adjoint_array[m],
                    primal_reconstruction(
                        temperature_array,
                        l,
                        gauss_2,
                        microtimestep_thermal_adjust,
                    )
                    - linear_extrapolation(
                        temperature_array, m, gauss_2, microtimestep_thermal
                    ),
                    structure,
                    thermal,
                    param,
                )
            )
            lhs += (
                0.5
                * time_step_size
                * form_structure_thermal_adjoint(
                    linear_extrapolation(
                        velocity_array, m, gauss_1, microtimestep_thermal
                    ),
                    linear_extrapolation(
                        displacement_array,
                        m,
                        gauss_1,
                        microtimestep_thermal,
                    ),
                    linear_extrapolation(
                        temperature_array,
                        m,
                        gauss_1,
                        microtimestep_thermal,
                    ),
                    velocity_adjoint_array[m],
                    displacement_adjoint_array[m],
                    primal_reconstruction(
                        temperature_array,
                        l,
                        gauss_1,
                        microtimestep_thermal_adjust,
                    )
                    - linear_extrapolation(
                        temperature_array, m, gauss_1, microtimestep_thermal
                    ),
                    structure,
                    thermal,
                    param,
                )
            )
            lhs += (
                0.5
                * time_step_size
                * form_structure_thermal_adjoint(
                    linear_extrapolation(
                        velocity_array, m, gauss_2, microtimestep_thermal
                    ),
                    linear_extrapolation(
                        displacement_array,
                        m,
                        gauss_2,
                        microtimestep_thermal,
                    ),
                    linear_extrapolation(
                        temperature_array,
                        m,
                        gauss_2,
                        microtimestep_thermal,
                    ),
                    velocity_adjoint_array[m],
                    displacement_adjoint_array[m],
                    primal_reconstruction(
                        temperature_array,
                        l,
                        gauss_2,
                        microtimestep_thermal_adjust,
                    )
                    - linear_extrapolation(
                        temperature_array, m, gauss_2, microtimestep_thermal
                    ),
                    structure,
                    thermal,
                    param,
                )
            )
            rhs += (
                0.5
                * time_step_size
                * 2.0
                * param.KAPPA
                * param.GOAL_FUNCTIONAL_THERMAL
                * dot(
                    linear_extrapolation(
                        temperature_array,
                        m,
                        gauss_1,
                        microtimestep_thermal,
                    ),
                    primal_reconstruction(
                        temperature_array,
                        l,
                        gauss_1,
                        microtimestep_thermal_adjust,
                    )
                    - linear_extrapolation(
                        temperature_array,
                        m,
                        gauss_1,
                        microtimestep_thermal,
                    ),
                )
                * thermal.dx
            )
            rhs += (
                0.5
                * time_step_size
                * 2.0
                * param.KAPPA
                * param.GOAL_FUNCTIONAL_THERMAL
                * dot(
                    linear_extrapolation(
                        temperature_array,
                        m,
                        gauss_2,
                        microtimestep_thermal,
                    ),
                    primal_reconstruction(
                        temperature_array,
                        l,
                        gauss_2,
                        microtimestep_thermal_adjust,
                    )
                    - linear_extrapolation(
                        temperature_array,
                        m,
                        gauss_2,
                        microtimestep_thermal,
                    ),
                )
                * thermal.dx
            )
            rhs += (
                0.5
                * time_step_size
                * zeta_adjoint(
                    primal_reconstruction(
                        temperature_array,
                        l,
                        gauss_1,
                        microtimestep_thermal_adjust,
                    )
                    - linear_extrapolation(
                        temperature_array, m, gauss_1, microtimestep_thermal
                    ),
                    param,
                )
                * inner(
                    epsilon(
                        linear_extrapolation(
                            displacement_array,
                            m,
                            gauss_1,
                            microtimestep_thermal,
                        )
                    ),
                    epsilon(
                        linear_extrapolation(
                            displacement_array,
                            m,
                            gauss_1,
                            microtimestep_thermal,
                        )
                    ),
                )
                * thermal.dx
            )
            rhs += (
                0.5
                * time_step_size
                * zeta_adjoint(
                    primal_reconstruction(
                        temperature_array,
                        l,
                        gauss_2,
                        microtimestep_thermal_adjust,
                    )
                    - linear_extrapolation(
                        temperature_array, m, gauss_2, microtimestep_thermal
                    ),
                    param,
                )
                * inner(
                    epsilon(
                        linear_extrapolation(
                            displacement_array,
                            m,
                            gauss_2,
                            microtimestep_thermal,
                        )
                    ),
                    epsilon(
                        linear_extrapolation(
                            displacement_array,
                            m,
                            gauss_2,
                            microtimestep_thermal,
                        )
                    ),
                )
                * thermal.dx
            )

            residuals.append(assemble(0.5 * rhs - 0.5 * lhs))
            m += 1
            microtimestep_thermal_before = microtimestep_thermal_before.after
            microtimestep_thermal = microtimestep_thermal.after

        macrotimestep_thermal = macrotimestep_thermal.after
    if param.PARTIAL_COMPUTE is not None:
        if param.PARTIAL_COMPUTE[0] != 0:
            residuals.append(0.0)

    return residuals
