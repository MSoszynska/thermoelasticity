def refine_time(
    primal_structure,
    primal_thermal,
    adjoint_structure,
    adjoint_thermal,
    structure_timeline,
    thermal_timeline,
):

    # Find intervals to refine
    structure_residuals = [
        abs(x[0] + x[1]) for x in zip(primal_structure, adjoint_structure)
    ]
    thermal_residuals = [
        abs(x[0] + x[1]) for x in zip(primal_thermal, adjoint_thermal)
    ]
    average = (
        2.0 * sum([abs(x) for x in structure_residuals])
        + 2.0 * sum([abs(x) for x in thermal_residuals])
    ) / (len(structure_residuals) + len(thermal_residuals))
    structure_refinements = []
    thermal_refinements = []
    for i in range(len(structure_residuals)):

        if structure_residuals[i] > average:

            structure_refinements.append(True)
        else:

            structure_refinements.append(False)

    for i in range(len(thermal_residuals)):

        if thermal_residuals[i] > average:

            thermal_refinements.append(True)
        else:

            thermal_refinements.append(False)

    # Adjust refinement array to preserve patch structure
    for i in range(len(structure_residuals)):

        if structure_refinements[i]:

            if i % 2 == 0:

                structure_refinements[i + 1] = True
            else:

                structure_refinements[i - 1] = True

    for i in range(len(thermal_residuals)):

        if thermal_refinements[i]:

            if i % 2 == 0:

                thermal_refinements[i + 1] = True
            else:

                thermal_refinements[i - 1] = True

    # Save refinement arrays
    structure_size = structure_timeline.size_global - structure_timeline.size
    thermal_size = thermal_timeline.size_global - thermal_timeline.size
    structure_refinements_txt = open(
        f"structure_{structure_size}-{thermal_size}_refinements.txt", "a"
    )
    thermal_refinements_txt = open(
        f"thermal_{structure_size}-{thermal_size}_refinements.txt", "a"
    )
    [
        structure_refinements_txt.write(str(int(x)))
        for x in structure_refinements
    ]
    [thermal_refinements_txt.write(str(int(x))) for x in thermal_refinements]
    structure_refinements_txt.close()
    thermal_refinements_txt.close()
