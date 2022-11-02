from fenics import (
    dot,
    grad,
    Expression,
    project,
    Constant,
    inner,
    div,
    Identity,
    interpolate,
    Function,
    assign,
    assemble,
    derivative,
    det,
    tr,
    inv,
)
from math import sqrt
from parameters import Parameters
from spaces import Space
from time_structure import MicroTimeStep

# Define coefficients of 2-point Gaussian quadrature
def gausss(microtimestep, microtimestep_before=None):

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


# Define goal functional
def goal_functional(
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
    primal_solutions,
    local_counter,
    param,
    space,
    initial=False,
):
    if space.name == "structure":
        displacement_array = primal_solutions.displacement_structure
        temperature_array = primal_solutions.temperature_structure
        time_array = primal_solutions.time_structure
    else:
        displacement_array = primal_solutions.displacement_thermal
        temperature_array = primal_solutions.temperature_thermal
        time_array = primal_solutions.time_thermal
    displacement_array_size = len(displacement_array)
    temperature_array_size = len(temperature_array)
    displacement_function_before = displacement_array[
        displacement_array_size - local_counter - 2
    ]
    temperature_function_before = temperature_array[
        temperature_array_size - local_counter - 2
    ]
    time_before = time_array[len(time_array) - local_counter - 2]

    if not initial:

        displacement_function = displacement_array[
            displacement_array_size - local_counter - 1
        ]
        temperature_function = temperature_array[
            temperature_array_size - local_counter - 1
        ]
        time = time_array[len(time_array) - local_counter - 1]

        def linear_extrapolation(time_gauss):
            if space.name == "structure":
                function = zeta(temperature_function, param) * epsilon(
                    displacement_function
                )
                function_before = zeta(
                    temperature_function_before, param
                ) * epsilon(displacement_function_before)
            else:
                function = inner(
                    epsilon(displacement_function),
                    epsilon(displacement_function),
                )
                function_before = inner(
                    epsilon(displacement_function_before),
                    epsilon(displacement_function_before),
                )
            return (function - function_before) / (
                time - time_before
            ) * time_gauss + (
                function_before * time - function * time_before
            ) / (
                time - time_before
            )

        time_gauss_1, time_gauss_2 = gausss(
            microtimestep, microtimestep_before
        )

        first_value = 0.5 * linear_extrapolation(time_gauss_1) * (
            -time_gauss_1 + time
        ) + 0.5 * linear_extrapolation(time_gauss_2) * (-time_gauss_2 + time)
        second_value = 0.5 * linear_extrapolation(time_gauss_1) * (
            time_gauss_1 - time_before
        ) + 0.5 * linear_extrapolation(time_gauss_2) * (
            time_gauss_2 - time_before
        )
    else:
        if space.name == "structure":
            first_value = 0.0 * epsilon(displacement_function_before)
            second_value = 0.0 * epsilon(displacement_function_before)
        else:
            first_value = 0.0 * inner(
                epsilon(displacement_function_before),
                epsilon(displacement_function_before),
            )
            second_value = 0.0 * inner(
                epsilon(displacement_function_before),
                epsilon(displacement_function_before),
            )
    return [
        first_value,
        second_value,
    ]


# Define variational problem
class Problem:
    def __init__(
        self,
        bilinear_form,
        functional,
        functional_interface=None,
    ):
        self.bilinear_form = bilinear_form
        self.functional = functional
        self.functional_interface = functional_interface


# Define material parameters
def E(temperature_primal, param):
    return param.E_INITIAL * (1.0 - param.BETA * temperature_primal)


def E_adjoint(temperature_adjoint, param):
    return -param.E_INITIAL * param.BETA * temperature_adjoint


def zeta(temperature_primal, param):
    return (param.NU * E(temperature_primal, param)) / (
        (1.0 + param.NU) * (1.0 - 2.0 * param.NU)
    )


def zeta_adjoint(temperature_adjoint, param):
    return (param.NU * E_adjoint(temperature_adjoint, param)) / (
        (1.0 + param.NU) * (1.0 - 2.0 * param.NU)
    )


def mu(temperature_primal, param):
    return E(temperature_primal, param) / (2.0 + 2.0 * param.NU)


def mu_adjoint(temperature_adjoint, param):
    return E_adjoint(temperature_adjoint, param) / (2.0 + 2.0 * param.NU)


# Define strain
def epsilon(displacement):
    return 0.5 * (grad(displacement) + grad(displacement).T)


# Define structure stress
def sigma_structure(displacement_primal, temperature_primal, structure, param):
    dimension = structure.dimension[0]
    return zeta(temperature_primal, param) * tr(
        epsilon(displacement_primal)
    ) * Identity(dimension) + 2.0 * mu(temperature_primal, param) * epsilon(
        displacement_primal
    )


def sigma_structure_adjoint(
    temperature_primal, displacement_adjoint, structure, param
):
    dimension = structure.dimension[0]
    return zeta(temperature_primal, param) * tr(
        epsilon(displacement_adjoint)
    ) * Identity(dimension) + 2.0 * mu(temperature_primal, param) * epsilon(
        displacement_adjoint
    )


def sigma_thermal_adjoint(
    displacement_primal, temperature_adjoint, structure, param
):
    dimension = structure.dimension[0]
    return zeta_adjoint(temperature_adjoint, param) * tr(
        epsilon(displacement_primal)
    ) * Identity(dimension) + 2.0 * mu_adjoint(
        temperature_adjoint, param
    ) * epsilon(
        displacement_primal
    )


def external_force(time):

    # time_leftover = time - int(time)
    output = False
    if time <= 0.1:
        output = True
    elif 2.0 <= time and time <= 2.1:
        output = True
    if output:
        function = Expression(
            (
                "2000.0 * cos(pi * 10 * time - 0.5 * pi) * cos(pi * 10 * time - 0.5 * pi)",
                "0.0",
            ),
            time=time,
            degree=2,
        )
    else:
        function = Expression(("0.0", "0.0"), degree=2)

    # function = Expression(("500.0 * cos(pi * time - 0.5 * pi) * cos(pi * time - 0.5 * pi)", "0.0"),
    #                      time=time, degree=2)

    return function


# Define variational forms of the primal structure subproblem
def form_structure(
    velocity_primal,
    displacement_primal,
    temperature_primal,
    first_test_function,
    second_test_function,
    structure: Space,
    param: Parameters,
):

    return (
        inner(
            sigma_structure(
                displacement_primal, temperature_primal, structure, param
            ),
            grad(first_test_function),
        )
        * structure.dx
        - dot(velocity_primal, second_test_function) * structure.dx
        + param.DELTA
        * inner(grad(velocity_primal), grad(first_test_function))
        * structure.dx
    )


def bilinear_form_structure(
    velocity_primal,
    displacement_primal,
    temperature_primal,
    first_test_function,
    second_test_function,
    structure: Space,
    thermal: Space,
    param: Parameters,
    time_step_size,
):

    return (
        param.RHO * dot(velocity_primal, first_test_function) * structure.dx
        + dot(displacement_primal, second_test_function) * structure.dx
        + param.THETA
        * time_step_size
        * form_structure(
            velocity_primal,
            displacement_primal,
            temperature_primal,
            first_test_function,
            second_test_function,
            structure,
            param,
        )
    )


def functional_structure(
    velocity_primal,
    displacement_primal,
    temperature_primal,
    velocity_primal_before,
    displacement_primal_before,
    temperature_primal_before,
    first_test_function,
    second_test_function,
    structure: Space,
    thermal: Space,
    param: Parameters,
    time,
    time_before,
    time_step_size,
):

    return (
        param.RHO
        * dot(velocity_primal_before, first_test_function)
        * structure.dx
        + dot(displacement_primal_before, second_test_function) * structure.dx
        - (1.0 - param.THETA)
        * time_step_size
        * form_structure(
            velocity_primal_before,
            displacement_primal_before,
            temperature_primal_before,
            first_test_function,
            second_test_function,
            structure,
            param,
        )
        + (1.0 - param.THETA)
        * time_step_size
        * param.RHO
        * dot(external_force(time_before), first_test_function)
        * structure.dx
        + param.THETA
        * time_step_size
        * param.RHO
        * dot(external_force(time), first_test_function)
        * structure.dx
    )


# Define variational forms of the adjoint structure subproblem
def form_structure_adjoint(
    velocity_primal,
    displacement_primal,
    temperature_primal,
    velocity_adjoint,
    displacement_adjoint,
    first_test_function,
    second_test_function,
    structure: Space,
    thermal: Space,
    param: Parameters,
):
    sigma_adjoint = sigma_structure_adjoint(
        temperature_primal, second_test_function, structure, param
    )
    form_adjoint = (
        inner(sigma_adjoint, grad(velocity_adjoint)) * structure.dx
        - dot(first_test_function, displacement_adjoint) * structure.dx
        + param.DELTA
        * inner(grad(first_test_function), grad(velocity_adjoint))
        * structure.dx
    )

    return form_adjoint


def bilinear_form_structure_adjoint(
    velocity_primal,
    displacement_primal,
    temperature_primal,
    velocity_adjoint,
    displacement_adjoint,
    temperature_adjoint,
    first_test_function,
    second_test_function,
    structure: Space,
    thermal: Space,
    param: Parameters,
    time_step_size,
    microtimestep: MicroTimeStep,
):

    return (
        param.RHO * dot(first_test_function, velocity_adjoint) * structure.dx
        + dot(second_test_function, displacement_adjoint) * structure.dx
        + param.THETA
        * time_step_size
        * form_structure_adjoint(
            velocity_primal,
            displacement_primal,
            temperature_primal,
            velocity_adjoint,
            displacement_adjoint,
            first_test_function,
            second_test_function,
            structure,
            thermal,
            param,
        )
    )


def functional_structure_adjoint(
    velocity_primal,
    displacement_primal,
    temperature_primal,
    velocity_primal_after,
    displacement_primal_after,
    temperature_primal_after,
    velocity_adjoint,
    displacement_adjoint,
    temperature_adjoint,
    velocity_adjoint_after,
    displacement_adjoint_after,
    temperature_adjoint_after,
    first_test_function,
    second_test_function,
    structure: Space,
    thermal: Space,
    param: Parameters,
    time_step_size,
    time_step_size_after,
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
    microtimestep_after: MicroTimeStep,
    primal_solutions,
    local_counter,
    initial,
):
    return (
        param.RHO
        * dot(first_test_function, velocity_adjoint_after)
        * structure.dx
        + dot(second_test_function, displacement_adjoint_after) * structure.dx
        - (1.0 - param.THETA)
        * time_step_size_after
        * form_structure_adjoint(
            velocity_primal,
            displacement_primal,
            temperature_primal,
            velocity_adjoint_after,
            displacement_adjoint_after,
            first_test_function,
            second_test_function,
            structure,
            thermal,
            param,
        )
        + 2.0
        * inner(
            goal_functional(
                microtimestep_before,
                microtimestep,
                primal_solutions,
                local_counter,
                param,
                structure,
            )[1],
            epsilon(second_test_function),
        )
        * structure.dx
        + 2.0
        * inner(
            goal_functional(
                microtimestep,
                microtimestep_after,
                primal_solutions,
                local_counter - 1,
                param,
                structure,
                initial,
            )[0],
            epsilon(second_test_function),
        )
        * structure.dx
        + param.THETA
        * time_step_size
        * param.ALPHA
        * dot(div(second_test_function), temperature_adjoint)
        * structure.dx
        + (1.0 - param.THETA)
        * time_step_size_after
        * param.ALPHA
        * dot(div(second_test_function), temperature_adjoint_after)
        * structure.dx
    )


# Define variational forms of the primal thermal subproblem
def form_thermal(
    temperature_primal,
    first_test_function,
    structure: Space,
    thermal: Space,
    param: Parameters,
):

    return (
        param.KAPPA
        * inner(
            grad(temperature_primal),
            grad(first_test_function),
        )
        * thermal.dx
    )


def bilinear_form_thermal(
    temperature_primal,
    first_test_function,
    structure: Space,
    thermal: Space,
    param: Parameters,
    time_step_size,
):

    return param.C * param.RHO * dot(
        temperature_primal, first_test_function
    ) * thermal.dx + param.THETA * time_step_size * form_thermal(
        temperature_primal,
        first_test_function,
        structure,
        thermal,
        param,
    )


def functional_thermal(
    velocity_primal,
    displacement_primal,
    temperature_primal,
    velocity_primal_before,
    displacement_primal_before,
    temperature_primal_before,
    first_test_function,
    structure: Space,
    thermal: Space,
    param: Parameters,
    time_step_size,
):

    return (
        param.C
        * param.RHO
        * dot(temperature_primal_before, first_test_function)
        * thermal.dx
        - (1.0 - param.THETA)
        * time_step_size
        * form_thermal(
            temperature_primal_before,
            first_test_function,
            structure,
            thermal,
            param,
        )
        + param.THETA
        * time_step_size
        * param.ALPHA
        * dot(div(displacement_primal), first_test_function)
        * thermal.dx
        + (1.0 - param.THETA)
        * time_step_size
        * param.ALPHA
        * dot(div(displacement_primal_before), first_test_function)
        * thermal.dx
    )


# Define variational forms of the adjoint thermal subproblem
def form_structure_thermal_adjoint(
    velocity_primal,
    displacement_primal,
    temperature_primal,
    velocity_adjoint,
    displacement_adjoint,
    first_test_function,
    structure: Space,
    thermal: Space,
    param: Parameters,
):
    sigma_adjoint = sigma_thermal_adjoint(
        displacement_primal, first_test_function, structure, param
    )
    form_adjoint = inner(sigma_adjoint, grad(velocity_adjoint)) * thermal.dx
    return form_adjoint


def form_thermal_adjoint(
    temperature_primal,
    temperature_adjoint,
    first_test_function,
    structure: Space,
    thermal: Space,
    param: Parameters,
):
    form_adjoint = (
        param.KAPPA
        * inner(
            grad(first_test_function),
            grad(temperature_adjoint),
        )
        * thermal.dx
    )
    return form_adjoint


def bilinear_form_thermal_adjoint(
    velocity_primal,
    displacement_primal,
    temperature_primal,
    temperature_adjoint,
    first_test_function,
    structure: Space,
    thermal: Space,
    param: Parameters,
    time_step_size,
    microtimestep: MicroTimeStep,
):

    return param.C * param.RHO * dot(
        first_test_function, temperature_adjoint
    ) * thermal.dx + param.THETA * time_step_size * form_thermal_adjoint(
        temperature_primal,
        temperature_adjoint,
        first_test_function,
        structure,
        thermal,
        param,
    )


def functional_thermal_adjoint(
    velocity_primal,
    displacement_primal,
    temperature_primal,
    velocity_primal_after,
    displacement_primal_after,
    temperature_primal_after,
    velocity_adjoint,
    displacement_adjoint,
    temperature_adjoint,
    velocity_adjoint_after,
    displacement_adjoint_after,
    temperature_adjoint_after,
    first_test_function,
    structure: Space,
    thermal: Space,
    param: Parameters,
    time_step_size,
    time_step_size_after,
    microtimestep_before: MicroTimeStep,
    microtimestep: MicroTimeStep,
    microtimestep_after: MicroTimeStep,
    primal_solutions,
    local_counter,
    initial,
):

    dimension = displacement_adjoint.geometric_dimension()
    if initial:
        temperature_adjoint_after = Function(thermal.function_space_split[0])
    temperature_primal = microtimestep.functions["primal_temperature"]

    return (
        param.C
        * param.RHO
        * dot(first_test_function, temperature_adjoint_after)
        * thermal.dx
        - (1.0 - param.THETA)
        * time_step_size_after
        * form_thermal_adjoint(
            temperature_primal,
            temperature_adjoint_after,
            first_test_function,
            structure,
            thermal,
            param,
        )
        - param.THETA
        * time_step_size
        * form_structure_thermal_adjoint(
            velocity_primal,
            displacement_primal,
            temperature_primal,
            velocity_adjoint,
            displacement_adjoint,
            first_test_function,
            structure,
            thermal,
            param,
        )
        - (1.0 - param.THETA)
        * time_step_size_after
        * form_structure_thermal_adjoint(
            velocity_primal,
            displacement_primal,
            temperature_primal,
            velocity_adjoint_after,
            displacement_adjoint_after,
            first_test_function,
            structure,
            thermal,
            param,
        )
        + zeta_adjoint(first_test_function, param)
        * goal_functional(
            microtimestep_before,
            microtimestep,
            primal_solutions,
            local_counter,
            param,
            thermal,
        )[1]
        * thermal.dx
        + zeta_adjoint(first_test_function, param)
        * goal_functional(
            microtimestep,
            microtimestep_after,
            primal_solutions,
            local_counter - 1,
            param,
            thermal,
            initial,
        )[0]
        * thermal.dx
    )
