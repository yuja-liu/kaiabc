using DifferentialEquations
using Plots
include("src/Models.jl")
include("src/Params.jl")

# initial values and some params
U0 = 3.5
X0 = [U0, 0, 0, 0, 0, 0]
A0 = 1.0
fATP = 1.0

p = [kUTA, kTU, kTUA, kTDA, kDT, kDTA, kDS, kDSA, kSDA, kSU, kSUA, kUSA, kCIhyd,
     KA, A0, fATP, N, M]

tspan = (0.0, 100.0)

prob = ODEProblem(kaiabc_phong!, X0, tspan, p)
sol = solve(prob, saveat=0.1)

plot(sol)
