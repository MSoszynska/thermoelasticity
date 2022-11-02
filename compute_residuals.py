from error_estimate import (
    primal_residual_structure,
    primal_residual_thermal,
    adjoint_residual_structure,
    adjoint_residual_thermal,
    goal_functional_structure,
)
from parameters import Parameters
from spaces import Space
from time_structure import TimeLine


def compute_residuals(
    structure: Space,
    thermal: Space,
    param: Parameters,
    structure_timeline: TimeLine,
    thermal_timeline: TimeLine,
):

    # Create text file
    residuals_txt = open("residuals.txt", "a")

    # Compute residuals
    primal_structure = primal_residual_structure(
        structure,
        thermal,
        structure_timeline,
        thermal_timeline,
        param,
    )
    print(
        f"Primal residual for the structure subproblem: "
        f"{sum(primal_structure)}"
    )
    [print(x) for x in primal_structure]
    residuals_txt.write(
        f"Primal residual for the structure subproblem: "
        f"{sum(primal_structure)} \r\n"
    )
    primal_thermal = primal_residual_thermal(
        thermal,
        structure,
        thermal_timeline,
        structure_timeline,
        param,
    )
    print(
        f"Primal residual for the thermal subproblem: "
        f"{sum(primal_thermal)}"
    )
    [print(x) for x in primal_thermal]
    residuals_txt.write(
        f"Primal residual for the thermal subproblem: "
        f"{sum(primal_thermal)} \r\n"
    )
    adjoint_structure = adjoint_residual_structure(
        structure,
        thermal,
        structure_timeline,
        thermal_timeline,
        param,
    )
    print(
        f"Adjoint residual for the structure subproblem: "
        f"{sum(adjoint_structure)}"
    )
    [print(x) for x in adjoint_structure]
    residuals_txt.write(
        f"Adjoint residual for the structure subproblem: "
        f"{sum(adjoint_structure)} \r\n"
    )
    adjoint_thermal = adjoint_residual_thermal(
        thermal,
        structure,
        thermal_timeline,
        structure_timeline,
        param,
    )
    print(
        f"Adjoint residual for the thermal subproblem: "
        f"{sum(adjoint_thermal)}"
    )
    [print(x) for x in adjoint_thermal]
    residuals_txt.write(
        f"Adjoint residual for the thermal subproblem: "
        f"{sum(adjoint_thermal)} \r\n"
    )

    # Compute goal functional
    goal_functional = goal_functional_structure(
        structure,
        structure_timeline,
        thermal,
        thermal_timeline,
        param,
    )

    print(f"Value of goal functional: {sum(goal_functional)}")
    residuals_txt.write(
        f"Value of goal functional: {sum(goal_functional)} \r\n"
    )
    residuals_txt.close()

    structure_residual = 0
    for i in range(len(primal_structure)):
        structure_residual += abs(primal_structure[i] + adjoint_structure[i])
    print(f"Value of structure residual: {structure_residual}")

    thermal_residual = 0
    for i in range(len(primal_thermal)):
        thermal_residual += abs(primal_thermal[i] + adjoint_thermal[i])
    print(f"Value of thermal residual: {thermal_residual}")

    partial_residual_structure = open("partial_residual_structure.txt", "a+")
    partial_residual_thermal = open("partial_residual_thermal.txt", "a+")
    for i in range(len(primal_structure)):
        partial_residual_structure.write(
            f"{abs(primal_structure[i] + adjoint_structure[i])}\r\n"
        )
    for i in range(len(primal_thermal)):
        partial_residual_thermal.write(
            f"{abs(primal_thermal[i] + adjoint_thermal[i])}\r\n"
        )
    partial_residual_structure.close()
    partial_residual_thermal.close()

    partial_residual_structure = open("partial_residual_structure.txt", "r")
    partial_residual_thermal = open("partial_residual_thermal.txt", "r")
    residual_structure_test = []
    for x in partial_residual_structure.read().splitlines():
        residual_structure_test.append(float(x))
    residual_thermal_test = []
    for x in partial_residual_thermal.read().splitlines():
        residual_thermal_test.append(float(x))
    partial_residual_structure.close()
    partial_residual_thermal.close()

    return [
        primal_structure,
        primal_thermal,
        adjoint_structure,
        adjoint_thermal,
        sum(goal_functional),
    ]
