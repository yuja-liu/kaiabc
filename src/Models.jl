using LinearAlgebra

"""
A model described in Phong et al. 2013. PNAS. that builds in the invariant
ATP hydrolysis rate in the CI ring.
Potential choices of implementation (parameter values,
rate-limiting steps, etc.) made by Chris Chi.
"""
function kaiabc_phong!(dX, X, p, t)

    # U: unphosphorylated. T: only T432 phosphorylated. S: only S431 phosphorylated
    # D: doubly phosphorylated. DB: D KaiC bound with KaiB. SB: S KaiC bound with KaiB
    U, T, D, S, DB, SB = X

    # k's are reaction constants (here all are 1st-order), e.g., 
    # kTU is the per reactant rate transfering from T to U
    # kCIhyd: ATP hydrolysis rate of the CI ring
    # KA: [KaiA] that activates kinase activity to half maximum
    # A0: initial [KaiA]
    # fATP = [ATP] / ([ATP] + [ADP])
    # N: # A sequestered per D. M: # A sequestered per S
    kUTA, kTU, kTUA, kTDA, kDT, kDTA, kDS, kDSA, kSDA, kSU, kSUA, kUSA, kCIhyd,
    KA, A0, fATP, N, M = p

    hill(x) = x/(x + KA)

    # KaiA sequestration
    A = max(0, A0 - N*DB - M*SB)

    #           U,      T,      D,      S,      DB,     SB
    dephos_r  = [0      kTU     0       kSU     0       kSU
                 0      0       kDT     0       kDT     0
                 0      0       0       0       0       0
                 0      0       kDS     0       0       0
                 0      0       0       0       0       0
                 0      0       0       0       kDS     0] +
                [0      kTUA    0       kSUA    0       kSUA
                 0      0       kDTA    0       kDTA    0
                 0      0       0       0       0       0
                 0      0       kDSA    0       0       0
                 0      0       0       0       0       0
                 0      0       0       0       kDSA    0]*hill(A)
    
    #         U     T       D       S       DB      SB
    phos_r = [0     0       0       0       0       0
              kUTA  0       0       0       0       0
              0     kTDA    0       kSDA    0       0
              kUSA  0       0       0       0       0
              0     0       0       0       0       kSDA
              0     0       0       0       0       0]*hill(A)*fATP

    #        U      T       D       S       DB      SB
    hyd_r = [0      0       0       0       0       0
             0      0       0       0       0       0
             0      0       0       0       0       0
             0      0       0       0       0       0
             0      0       kCIhyd  0       0       0
             0      0       0       kCIhyd  0       0]

    r = dephos_r + phos_r + hyd_r

    # conservation
    r -= I(size(r)[1]) .* sum(r, dims=1)

    # (for AD) Zygote.Buffer doesn't support in-place broadcast .=
    # see https://discourse.julialang.org/t/how-to-use-initialize-zygote-buffer/87653
    dX[:] = r*X
    nothing
end



"""
Phong model with constant growth rate and constant KaiC expression rate.
"""
function kaiabc_growing!(dX, X, p, t)
    tau = p[end - 1]    # doubling time
    C_ss = p[end]    # steady-state value of total [C]
    p_rest = p[1:lastindex(p) - 2]
    
    kaiabc_phong!(dX, X, p_rest, t)
    
    # synthesis
    k_dilu = log(2)/tau
    k_syn = k_dilu*C_ss
    dX[1] += k_syn
    
    # dilution
    dX[:] -= k_dilu*X
    nothing
end

"""
Right-hand-side function of the KaiABC system with protein synthesis and dilution.
Protein production rate (a lumped term of transcription and translation) is assumed
to be regulated by the clock phase, which is proxyed by the ratio of KaiC phosphoforms
at a given time (where the molecular mechanism is termed TTFL, transcription-translation feedback loop).
Dilution rate is assumed to be independent of the clock phase.
"""
function kaiabc_growing_TTFL!(dX, X, p, t)
    tau = p[end - 3]
    C_ss = p[end - 2]
    fb = p[end - 1]
    p_fb = p[end]
    p_rest = p[1:lastindex(p) - 4]
    
    kaiabc_phong!(dX, X, p_rest, t)
    
    # synthesis
    k_dilu = log(2)/tau
    # protein synthesis rate is still proportional to growth rate
    k_syn_b = k_dilu*C_ss
    k_syn = k_syn_b*fb(X, p_fb)
    dX[1] += k_syn
    
    # dilution
    # assumed to still be first-order i.e. volume expansion rate
    # is independent of the clock. See Yang 2010 Science
    dX[:] -= k_dilu*X
    
    nothing
end