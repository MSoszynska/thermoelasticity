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
from parameters import Parameters
from spaces import Space
from time_structure import MicroTimeStep, MacroTimeStep, TimeLine
from math import sqrt

# Define coefficients of 2-point Gaussian quadrature
def gauss(microtimestep, microtimestep_before=None):

    t_new = microtimestep.point
    if microtimestep_before is None:
        t_old = microtimestep.before.point
        dt = microtimestep.before.dt
    else:
        t_old = microtimestep_before.point
        dt = microtimestep.point - microtimestep_before.point
    t_average = 0.5 * (t_old + t_new)
    return [
        dt / (2.0 * sqrt(3)) + t_average,
        -dt / (2.0 * sqrt(3)) + t_average,
    ]


# Copy solutions
def copy_list(
    timeline: TimeLine,
    function_name,
    param: Parameters,
    adjust_size,
):

    array = []
    if param.PARTIAL_COMPUTE is not None:
        if adjust_size:
            if param.PARTIAL_COMPUTE[0] != 0:
                starting_point = param.PARTIAL_COMPUTE[0] - 2
                size = param.PARTIAL_COMPUTE[1] + 2
            else:
                starting_point = param.PARTIAL_COMPUTE[0]
                size = param.PARTIAL_COMPUTE[1]
        else:
            starting_point = param.PARTIAL_COMPUTE[0]
            size = param.PARTIAL_COMPUTE[1]
        macrotimestep_adjust = timeline.head
        for i in range(starting_point):
            macrotimestep_adjust = macrotimestep_adjust.after
        macrotimestep = macrotimestep_adjust
        global_size = size
    else:
        macrotimestep = timeline.head
        global_size = timeline.size
    for n in range(global_size):

        microtimestep = macrotimestep.head
        local_size = macrotimestep.size
        for m in range(local_size):

            function = microtimestep.functions[function_name]
            if not microtimestep.after is None:

                array.append(function.copy(deepcopy=True))

            if (microtimestep.after is None) and (n == global_size - 1):

                array.append(function.copy(deepcopy=True))

            microtimestep = microtimestep.after

        macrotimestep = macrotimestep.after

    return array


# Extrapolate solutions
def extrapolate_list(
    space: Space,
    space_timeline: TimeLine,
    space_interface: Space,
    space_interface_timeline: TimeLine,
    function_name,
    subspace_index,
    param: Parameters,
    adjust_size,
):
    if param.PARTIAL_COMPUTE is not None:
        if adjust_size:
            if param.PARTIAL_COMPUTE[0] != 0:
                starting_point = param.PARTIAL_COMPUTE[0] - 2
                size = param.PARTIAL_COMPUTE[1] + 2
            else:
                starting_point = param.PARTIAL_COMPUTE[0]
                size = param.PARTIAL_COMPUTE[1]
        else:
            starting_point = param.PARTIAL_COMPUTE[0]
            size = param.PARTIAL_COMPUTE[1]
        space_macrotimestep_adjust = space_timeline.head
        space_interface_macrotimestep_adjust = space_interface_timeline.head
        for i in range(starting_point):
            space_macrotimestep_adjust = space_macrotimestep_adjust.after
            space_interface_macrotimestep_adjust = (
                space_interface_macrotimestep_adjust.after
            )
        space_macrotimestep = space_macrotimestep_adjust
        space_interface_macrotimestep = space_interface_macrotimestep_adjust
        global_size = size
    else:
        space_macrotimestep = space_timeline.head
        space_interface_macrotimestep = space_interface_timeline.head
        global_size = space_timeline.size
    array = []
    function = space_interface_macrotimestep.head.functions[function_name]
    array.append(function)
    for n in range(global_size):

        space_microtimestep = space_macrotimestep.head
        local_size = space_macrotimestep.size - 1
        for m in range(local_size):

            extrapolation_proportion = (
                space_macrotimestep.tail.point
                - space_microtimestep.after.point
            ) / space_macrotimestep.dt
            function_old = space_interface_macrotimestep.head.functions[
                function_name
            ]
            function_new = space_interface_macrotimestep.tail.functions[
                function_name
            ]
            array.append(
                project(
                    extrapolation_proportion * function_old
                    + (1.0 - extrapolation_proportion) * function_new,
                    space_interface.function_space_split[subspace_index],
                )
            )
            space_microtimestep = space_microtimestep.after

        space_macrotimestep = space_macrotimestep.after
        space_interface_macrotimestep = space_interface_macrotimestep.after

    return array


# Define linear extrapolation
def linear_extrapolation(array, m, time, microtimestep: MicroTimeStep):

    time_step_size = microtimestep.before.dt
    point = microtimestep.point

    return (array[m] - array[m - 1]) / time_step_size * time + (
        array[m - 1] * point - array[m] * (point - time_step_size)
    ) / time_step_size


# Define reconstruction of the primal problem
def primal_reconstruction(array, m, time, microtimestep: MicroTimeStep):

    time_step_size = microtimestep.before.dt
    point = microtimestep.point
    a = (array[m + 1] - 2 * array[m] + array[m - 1]) / (
        2.0 * time_step_size * time_step_size
    )
    b = (
        (time_step_size - 2.0 * point) * array[m + 1]
        + 4 * point * array[m]
        + (-time_step_size - 2.0 * point) * array[m - 1]
    ) / (2.0 * time_step_size * time_step_size)
    c = (
        (-time_step_size * point + point * point) * array[m + 1]
        + (2.0 * time_step_size * time_step_size - 2.0 * point * point)
        * array[m]
        + (time_step_size * point + point * point) * array[m - 1]
    ) / (2.0 * time_step_size * time_step_size)

    return a * time * time + b * time + c
    # return 0.0 * a


def primal_derivative(array, m, time, microtimestep: MicroTimeStep):

    time_step_size = microtimestep.before.dt
    point = microtimestep.point
    a = (array[m + 1] - 2 * array[m] + array[m - 1]) / (
        2.0 * time_step_size * time_step_size
    )
    b = (
        (time_step_size - 2.0 * point) * array[m + 1]
        + 4 * point * array[m]
        + (-time_step_size - 2.0 * point) * array[m - 1]
    ) / (2.0 * time_step_size * time_step_size)

    return 2.0 * a * time + b
    # return 0.0 * a


# Define reconstruction of the adjoint problem
def adjoint_reconstruction(array, m, time, microtimestep, macrotimestep):

    size = len(array) - 1
    if m == 1 or m == size:

        return array[m]
        # return 0.0 * array[m]

    else:

        if microtimestep.before.before is None:

            t_average_before = 0.5 * (
                microtimestep.before.point
                + macrotimestep.microtimestep_before.point
            )

        else:

            t_average_before = 0.5 * (
                microtimestep.before.point + microtimestep.before.before.point
            )

        if microtimestep.after is None:

            t_average_after = 0.5 * (
                microtimestep.point + macrotimestep.microtimestep_after.point
            )

        else:

            t_average_after = 0.5 * (
                microtimestep.point + microtimestep.after.point
            )

        return (time - t_average_before) / (
            t_average_after - t_average_before
        ) * array[m + 1] + (time - t_average_after) / (
            t_average_before - t_average_after
        ) * array[
            m - 1
        ]
        # return 0.0 * array[m - 1]


def time_array(timeline: TimeLine):

    result = []
    macrotimestep = timeline.head
    global_size = timeline.size
    for n in range(global_size):

        microtimestep = macrotimestep.head
        local_size = macrotimestep.size
        for m in range(local_size - 1):

            result.append(microtimestep.point)
            microtimestep = microtimestep.after

        macrotimestep = macrotimestep.after

    result.append(timeline.tail.tail.point)

    return result
