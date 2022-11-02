from fenics import Function, assign
from initial import Initial
from spaces import Space

# Define generic node
class Node:
    def __init__(self):

        self.after = None
        self.before = None

    # Define inserting a given element in the front
    def insert(self, element):

        if self.after is not None:

            element.after = self.after
            self.after.before = element

        self.after = element
        element.before = self


# Define generic two-way list
class List:
    def __init__(self):

        self.head = None
        self.tail = None

    # Compute size
    @property
    def size(self):

        result = 0
        element = self.head
        while element is not None:

            element = element.after
            result += 1

        return result


# Define single micro time-step
class MicroTimeStep(Node):
    def __init__(self, point):

        Node.__init__(self)
        self.point = point
        self.functions = {
            "primal_velocity": None,
            "primal_displacement": None,
            "primal_temperature": None,
            "adjoint_velocity": None,
            "adjoint_displacement": None,
            "adjoint_temperature": None,
        }

    # Define length of the micro time-step in relation to the node in front
    @property
    def dt(self):

        if self.after is None:

            return 0.0

        else:

            return self.after.point - self.point

    # Define refining an element by splitting it to two
    def refine(self):

        element = MicroTimeStep(self.point + 0.5 * self.dt)
        self.insert(element)


# Define macro time-step consisting of a list of micro time-steps
class MacroTimeStep(Node, List):
    def __init__(self):

        Node.__init__(self)
        List.__init__(self)

    # Define length of a macrotimestep
    @property
    def dt(self):

        return self.tail.point - self.head.point

    # Return neighbouring microtimesteps
    @property
    def microtimestep_before(self):

        if self.before is None:

            return None

        else:

            return self.before.tail.before

    @property
    def microtimestep_after(self):

        if self.after is None:

            return None

        else:

            return self.after.head.after

    # Create a macrotimestep consisting of equidistant microtimesteps
    def unify(self, final_time, time_step_size, size):

        self.head = MicroTimeStep(final_time)
        element = self.head
        for m in range(size):

            element.insert(
                MicroTimeStep(final_time + time_step_size * (m + 1))
            )
            element = element.after

        self.tail = element

    # Adjust position of the tail list after refinement of a microtimestep
    def refine(self, element: MicroTimeStep):

        element.refine()
        if self.tail.after is not None:

            self.tail = self.tail.after

    # Define splitting of a macrotimestep to two in a given microtimestep
    def split(self, element: MicroTimeStep):

        initial = MacroTimeStep()
        initial.head = MicroTimeStep(element.point)
        initial.tail = self.tail
        self.tail = element
        initial.head.after = element.after
        initial.head.after.before = initial.head
        initial.after = self.after
        if self.after is not None:

            initial.after.before = initial

        self.after = initial
        initial.before = self
        element.after = None


# Define timeline consisting of a list of macro time-steps
class TimeLine(List):
    def __init__(self):

        List.__init__(self)

    # Compute number of nodes in a timeline
    # (ends of inner macrotimesteps are computed twice)
    @property
    def size_global(self):

        result = 0
        element = self.head
        size = self.size
        for n in range(size):

            result += element.size
            element = element.after

        return result

    # Print content of a timeline
    def print(self):

        macrotimestep = self.head
        global_size = self.size
        for n in range(global_size):

            print(f"Macro time-step numeber: {n + 1}")
            print(f"Macro time-step size: {macrotimestep.size}")
            microtimestep = macrotimestep.head
            local_size = macrotimestep.size
            for m in range(local_size):

                print(
                    f"Node value: {microtimestep.point}, "
                    f"Micro time-step length: {microtimestep.dt}"
                )
                microtimestep = microtimestep.after

            print("\r\n")
            macrotimestep = macrotimestep.after

    # Create a macrotimestep consisting of equidistant microtimesteps
    def unify(self, time_step_size, local_size, global_size, initial_time):

        initial = MacroTimeStep()
        initial.unify(initial_time, time_step_size / local_size, local_size)
        self.head = initial
        element = self.head
        for n in range(global_size - 1):

            initial = MacroTimeStep()
            initial.unify(
                initial_time + time_step_size * (n + 1),
                time_step_size / local_size,
                local_size,
            )
            element.insert(initial)
            element = element.after

        self.tail = element

    # Given an array of bool values refine microtimesteps of a timeline
    def refine(self, array):

        macrotimestep = self.head
        global_size = self.size
        i = 0
        for n in range(global_size):

            microtimestep = macrotimestep.head
            local_size = macrotimestep.size - 1
            for m in range(local_size):

                if array[i]:

                    macrotimestep.refine(microtimestep)
                    microtimestep = microtimestep.after

                microtimestep = microtimestep.after
                i += 1

            macrotimestep = macrotimestep.after

    # Adjust position of a timeline list tail after splitting a macrotimestep
    def split(
        self, macrotimestep: MacroTimeStep, microtimestep: MicroTimeStep
    ):

        macrotimestep.split(microtimestep)
        if not self.tail.after is None:

            self.tail = self.tail.after

    # Load solutions to nodes
    def load(self, space: Space, space_name, adjoint):

        if not adjoint:
            if space.name == "structure":
                first_variable = Initial(
                    space_name,
                    "primal_velocity",
                    space.function_space_split[0],
                )
                second_variable = Initial(
                    space_name,
                    "primal_displacement",
                    space.function_space_split[1],
                )
            if space.name == "thermal":
                first_variable = Initial(
                    space_name,
                    "primal_temperature",
                    space.function_space_split[0],
                )
        else:
            if space.name == "structure":
                first_variable_adjoint = Initial(
                    space_name,
                    "adjoint_velocity",
                    space.function_space_split[0],
                )
                second_variable_adjoint = Initial(
                    space_name,
                    "adjoint_displacement",
                    space.function_space_split[1],
                )
            if space.name == "thermal":
                first_variable_adjoint = Initial(
                    space_name,
                    "adjoint_temperature",
                    space.function_space_split[0],
                )

        if adjoint:
            counter = 0
            macrotimestep = self.tail
            global_size = self.size
            for n in range(global_size):

                microtimestep = macrotimestep.tail
                local_size = macrotimestep.size
                for m in range(local_size):

                    if space.name == "structure":
                        microtimestep.functions[
                            "adjoint_velocity"
                        ] = first_variable_adjoint.load(counter)
                        microtimestep.functions[
                            "adjoint_displacement"
                        ] = second_variable_adjoint.load(counter)
                    if space.name == "thermal":
                        microtimestep.functions[
                            "adjoint_temperature"
                        ] = first_variable_adjoint.load(counter)
                    if microtimestep.before is not None:

                        counter += 1

                    microtimestep = microtimestep.before

                macrotimestep = macrotimestep.before

        else:

            counter = 0
            macrotimestep = self.head
            global_size = self.size
            for n in range(global_size):

                microtimestep = macrotimestep.head
                local_size = macrotimestep.size
                for m in range(local_size):

                    if space.name == "structure":
                        microtimestep.functions[
                            "primal_velocity"
                        ] = first_variable.load(counter)
                        microtimestep.functions[
                            "primal_displacement"
                        ] = second_variable.load(counter)
                    if space.name == "thermal":
                        microtimestep.functions[
                            "primal_temperature"
                        ] = first_variable.load(counter)

                    if microtimestep.after is not None:
                        counter += 1

                    microtimestep = microtimestep.after

                macrotimestep = macrotimestep.after


# Split macrotimesteps of timelines if their inner microtimesteps coincide
def split(structure_timeline: TimeLine, thermal_timeline: TimeLine):

    structure_macrotimestep = structure_timeline.head
    thermal_macrotimestep = thermal_timeline.head
    global_size = structure_timeline.size
    for n in range(global_size):

        structure_microtimestep = structure_macrotimestep.head.after
        thermal_microtimestep = thermal_macrotimestep.head.after
        stop = False
        while not stop:

            if (structure_microtimestep.after is None) or (
                thermal_microtimestep.after is None
            ):

                stop = True

            if (not stop) and (
                structure_microtimestep.point == thermal_microtimestep.point
            ):

                structure_timeline.split(
                    structure_macrotimestep, structure_microtimestep
                )
                thermal_timeline.split(
                    thermal_macrotimestep, thermal_microtimestep
                )
                structure_macrotimestep = structure_macrotimestep.after
                thermal_macrotimestep = thermal_macrotimestep.after
                stop = True

            if structure_microtimestep.point > thermal_microtimestep.point:

                thermal_microtimestep = thermal_microtimestep.after

            if structure_microtimestep.point < thermal_microtimestep.point:

                structure_microtimestep = structure_microtimestep.after

        structure_macrotimestep = structure_macrotimestep.after
        thermal_macrotimestep = thermal_macrotimestep.after
