from math import log

fsi1 = False
fsi2 = False
fsi3 = True
uniform_equal = False
uniform_refined = False
adaptive = False

primal_residual_fluid = []
primal_residual_solid = []
adjoint_residual_fluid = []
adjoint_residual_solid = []
functional_extrapolation = []
functional = []

if fsi1:
    functional_extrapolation.append(13.097046631988745)
    functional_extrapolation.append(13.10575041730394)
    functional_extrapolation.append(13.107817558001134)
    functional_extrapolation.append(13.108334740827614)
    functional_extrapolation.append(13.108464027051722)

if fsi2:
    functional_extrapolation.append(13.097046631988745)
    functional_extrapolation.append(13.10575041730394)
    functional_extrapolation.append(13.107817558001134)
    functional_extrapolation.append(13.108334740827614)
    functional_extrapolation.append(13.108464027051722)

if fsi3:
    functional_extrapolation.append(13.097046631988745)
    functional_extrapolation.append(13.10575041730394)
    functional_extrapolation.append(13.107817558001134)
    functional_extrapolation.append(13.108334740827614)
    functional_extrapolation.append(13.108464027051722)

if fsi1:

    primal_residual_fluid.append(0.001565136911773855)
    primal_residual_fluid.append(0.0005076859732370174)
    primal_residual_fluid.append(0.0001236155922457776)
    primal_residual_fluid.append(3.049132391527241e-05)
    primal_residual_fluid.append(7.5969182924639775e-06)

    primal_residual_solid.append(6.076103936431827e-06)
    primal_residual_solid.append(1.5237716911392764e-06)
    primal_residual_solid.append(3.803134005823879e-07)
    primal_residual_solid.append(9.5088522563495e-08)
    primal_residual_solid.append(2.3776529111276092e-08)

    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)

    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)

    functional.append(13.097046631988745)
    functional.append(13.10575041730394)
    functional.append(13.107817558001134)
    functional.append(13.108334740827614)
    functional.append(13.108464027051722)

if fsi2:
    primal_residual_fluid.append(0.00446975459187164)
    primal_residual_fluid.append(0.004741264665970095)
    primal_residual_fluid.append(0.0014548462033978976)
    primal_residual_fluid.append(0.0011204817533066477)
    primal_residual_fluid.append(0.002798779210041835)

    primal_residual_solid.append(6.398668200049408e-06)
    primal_residual_solid.append(8.96323930427012e-06)
    primal_residual_solid.append(8.290364448912692e-06)
    primal_residual_solid.append(7.3618991590540905e-06)
    primal_residual_solid.append(9.863826863014367e-06)

    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)

    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)

    functional.append(13.103276490144413)
    functional.append(13.106522731870093)
    functional.append(13.107148948301733)
    functional.append(13.107310741505836)
    functional.append(13.107928333470209)

if fsi3:
    primal_residual_fluid.append(0.000507702292499686)
    primal_residual_fluid.append(0.00012361601049469035)
    primal_residual_fluid.append(3.0491299702465637e-05)
    primal_residual_fluid.append(7.5969112708921574e-06)
    primal_residual_fluid.append(-0.010733144398890839)

    primal_residual_solid.append(6.084911666722659e-06)
    primal_residual_solid.append(1.5243974602895117e-06)
    primal_residual_solid.append(3.807066346258009e-07)
    primal_residual_solid.append(9.515234406726607e-08)
    primal_residual_solid.append(-3.694945661456657e-07)

    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)
    adjoint_residual_fluid.append(0.0)

    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)
    adjoint_residual_solid.append(0.0)

    functional.append(13.105752228457462)
    functional.append(13.107818019060723)
    functional.append(13.108334853482635)
    functional.append(13.108464055463497)
    functional.append(13.499984209520312)

# Perform extrapolation
print("Extrapolation")
J_exact = 0.0
J = functional_extrapolation
for i in range(len(J) - 2):

    print(f"Extrapolation of J{i + 1}, J{i + 2}, J{i + 3}")
    q = -log(abs((J[i + 1] - J[i + 2]) / (J[i] - J[i + 1]))) / log(2.0)
    print(f"Extrapolated order of convergence: {q}")
    C = pow(J[i] - J[i + 1], 2) / (J[i] - 2.0 * J[i + 1] + J[i + 2])
    print(f"Extrapolated constant: {C}")
    J_exact = (J[i] * J[i + 2] - J[i + 1] * J[i + 1]) / (
        J[i] - 2.0 * J[i + 1] + J[i + 2]
    )
    print(f"Extrapolated exact value of goal functional: {J_exact}")
# J_exact = 13.691283589090164

# Compute effectiveness
print("Effectivity")
J = functional
for i in range(len(J)):

    print(f"Effectivity of J{i + 1}")
    residual = (
        primal_residual_fluid[i]
        + primal_residual_solid[i]
        + adjoint_residual_fluid[i]
        + adjoint_residual_solid[i]
    )
    print(f"Overall residual: {residual}")
    print(f"Extrapolated error: {J_exact - J[i]}")
    effectivity = residual / (J_exact - J[i])
    print(f"Effectivity: {effectivity}")
