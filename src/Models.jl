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


using LinearAlgebra

"""
Implementing Chris Chi's Kai monomer model in Julia.
Chris's model explicitly models KaiA binding to CI and KaiA binding to CII-KaiB.
This is desired for modeling a potential change in KaiC CII-CI allostery as a function of temperature
"""
function kaiabc_chi!(dXdt, X, p, t)
    # p is a named tuple with the following mandatory elements
    # ln_ks: natural log of the absolute value of a list of parameters. See below
    # fATP = [ATP]/([ATP] + [ADP])
    # Atot: total [KaiA]
    # M: Number of KaiA molecules could bind to CI ringh (with KaiB) per KaiC monomer
    # N: Number of KaiA molecules could bind to CII ring per KaiC monomer
    ln_ks, fATP, Atot, M, N = p.ln_ks, p.fATP, p.Atot, p.M, p.N
    
    # kCIIAoffU set to be 1. Actual value is absorbed in the units of time
    kCIIAoffU, kTU, kSU, kTUA, kSUA, kCIIAoffT, kDT, kUTA, kDTA, kCIIAoffD, kRelD00,
        kTDA, kSDA, kRelDA0, kCIIAoffS, kDS, kRelS00, kUSA, kDSA, kRelSA0,
        kCIhydD00, kCIIAoffDA0, kCIAoffD0A, kCIhydDA0, kSDA0, kCIAoffDAA, kCIhydS00, 
        kDS00, kCIIAoffSA0, kCIAoffS0A, kCIhydSA0, kDSA0, kCIAoffSAA, kCIIAoffDAA, 
        kSDAA, kDS0A, kCIIAoffSAA, kDSAA, kCIIAonU, kCIIAonT, kCIIAonD, kCIIAonS, 
        kCIIAonD00, kCIIAonS00, kCIAonD00, kCIAonDA0, kCIIAonD0A, kCIAonS00, 
        kCIAonSA0, kCIIAonS0A = exp.([0.; ln_ks])
    
    # Stoichiometry: how many molecules of KaiA does one molecule of each species have? (row vector)
    #         1       2       3       4       5       6       7       8       9       10      11      12      13      14      15      16
    #         U       UA      T       TA      D       DA      S       SA      D00     DA0     S00     SA0     D0A     DAA     S0A     SAA
    nA_vec = [0.      N       0.      N       0.      N       0.      N       0.      N       0.      N       M       N + M   M       N + M] 
    
    # Expand X by conservation law of KaiC ([KaiC]tot is nondimensionalized to be 1)
    Xfull = [1. - sum(X); X]
    
    # [KaiA]free
    # !!Af can be negative!!
    Af = Atot - nA_vec⋅Xfull    # a true dot product: gives a scalar
    
    # Rate matrices. Contributions to products only (to reactants will be calculated by conservation law)
    # phosphorylation and dephosphorylation
    #           1         2         3         4         5         6         7         8         9         10        11        12        13        14        15        16     
    #           U         UA        T         TA        D         DA        S         SA        D00       DA0       S00       SA0       D0A       DAA       S0A       SAA       
    pdp_r =    [0.        0.        kTU       0.        0.        0.        kSU       0.        0.        0.        0.        0.        0.        0.        0.        0.    # 1 d(U)dt
                0.        0.        0.        kTUA      0.        0.        0.        kSUA      0.        0.        0.        0.        0.        0.        0.        0.    # 2 d(UA)dt
                0.        0.        0.        0.        kDT       0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.    # 3 d(T)dt
                0.        kUTA*fATP 0.        0.        0.        kDTA      0.        0.        0.        0.        0.        0.        0.        0.        0.        0.    # 4 d(TA)dt
                0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.    # 5 d(D)dt
                0.        0.        0.        kTDA*fATP 0.        0.        0.        kSDA*fATP 0.        0.        0.        0.        0.        0.        0.        0.    # 6 d(DA)dt
                0.        0.        0.        0.        kDS       0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.    # 7 d(S)dt
                0.        kUSA*fATP 0.        0.        0.        kDSA      0.        0.        0.        0.        0.        0.        0.        0.        0.        0.    # 8 d(SA)dt
                0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.    # 9 d(D00)dt
                0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        kSDA0     0.        0.        0.        0.    # 10 d(DA0)dt
                0.        0.        0.        0.        0.        0.        0.        0.        kDS00     0.        0.        0.        0.        0.        0.        0.    # 11 d(S00)dt
                0.        0.        0.        0.        0.        0.        0.        0.        0.        kDSA0     0.        0.        0.        0.        0.        0.    # 12 d(SA0)dt
                0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.    # 13 d(D0A)dt
                0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        kSDAA # 14 d(DAA)dt
                0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        kDS0A     0.        0.        0.    # 15 d(S0A)dt
                0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        kDSAA     0.        0.]   # 16 d(SAA)dt
    
    # Bimolecular binding involving KaiA
    #               1         2         3         4         5         6         7         8         9         10        11        12        13        14        15        16 
    #               U         UA        T         TA        D         DA        S         SA        D00       DA0       S00       SA0       D0A       DAA       S0A       SAA  
    bindA_r =   Af*[0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.    # 1 d(U)dt
                    kCIIAonU  0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.    # 2 d(UA)dt
                    0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.    # 3 d(T)dt
                    0.        0.        kCIIAonT  0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.    # 4 d(TA)dt
                    0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.    # 5 d(D)dt
                    0.        0.        0.        0.        kCIIAonD  0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.    # 6 d(DA)dt
                    0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.    # 7 d(S)dt
                    0.        0.        0.        0.        0.        0.        kCIIAonS  0.        0.        0.        0.        0.        0.        0.        0.        0.    # 8 d(SA)dt
                    0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.    # 9 d(D00)dt
                    0.        0.        0.        0.        0.        0.        0.        0.       kCIIAonD00 0.        0.        0.        0.        0.        0.        0.    # 10 d(DA0)dt
                    0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.    # 11 d(S00)dt
                    0.        0.        0.        0.        0.        0.        0.        0.        0.        0.       kCIIAonS00 0.        0.        0.        0.        0.    # 12 d(SA0)dt
                    0.        0.        0.        0.        0.        0.        0.        0.       kCIAonD00  0.        0.        0.        0.        0.        0.        0.    # 13 d(D0A)dt
                    0.        0.        0.        0.        0.        0.        0.        0.        0.        kCIAonDA0 0.        0.       kCIIAonD0A 0.        0.        0.    # 14 d(DAA)dt
                    0.        0.        0.        0.        0.        0.        0.        0.        0.        0.       kCIAonS00  0.        0.        0.        0.        0.    # 15 d(S0A)dt
                    0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        kCIAonSA0 0.        0.       kCIIAonS0A 0.]   # 16 d(SAA)dt
    
    # Binding or unbinding that doesn't involve KaiA
    #           1         2         3         4         5         6         7         8         9         10        11        12        13        14        15        16
    #           U         UA        T         TA        D         DA        S         SA        D00       DA0       S00       SA0       D0A       DAA       S0A       SAA  
    bunb_r =   [0.        kCIIAoffU 0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        # 1 d(U)dt
                0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        # 2 d(UA)dt
                0.        0.        0.        kCIIAoffT 0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        # 3 d(T)dt
                0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        # 4 d(TA)dt
                0.        0.        0.        0.        0.        kCIIAoffD 0.        0.        kRelD00   0.        0.        0.        0.        0.        0.        0.        # 5 d(D)dt
                0.        0.        0.        0.        0.        0.        0.        0.        0.      kRelDA0     0.        0.        0.        0.        0.        0.        # 6 d(DA)dt
                0.        0.        0.        0.        0.        0.        0.        kCIIAoffS 0.        0.        kRelS00   0.        0.        0.        0.        0.        # 7 d(S)dt
                0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.      kRelSA0     0.        0.        0.        0.        # 8 d(SA)dt
                0.        0.        0.        0.   kCIhydD00*fATP 0.        0.        0.        0.      kCIIAoffDA0 0.        0.     kCIAoffD0A   0.        0.        0.        # 9 d(D00)dt
                0.        0.        0.        0.        0.   kCIhydDA0*fATP 0.        0.        0.        0.        0.        0.        0.      kCIAoffDAA  0.        0.        # 10 d(DA0)dt
                0.        0.        0.        0.        0.        0.   kCIhydS00*fATP 0.        0.        0.        0.      kCIIAoffSA0 0.        0.       kCIAoffS0A 0.        # 11 d(S00)dt
                0.        0.        0.        0.        0.        0.        0.   kCIhydSA0*fATP 0.        0.        0.        0.        0.        0.        0.      kCIAoffSAA  # 12 d(SA0)dt
                0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.      kCIIAoffDAA 0.        0.        # 13 d(D0A)dt
                0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        # 14 d(DAA)dt
                0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.      kCIIAoffSAA # 15 d(S0A)dt
                0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.]       # 16 d(SAA)dt
    
    # Derivatives
    r = pdp_r + bindA_r + bunb_r
    # Conservation law of KaiC
    r -= I(length(Xfull)).*sum(r, dims = 1)
    
    dXdt_full = r*Xfull
    
    dXdt[:] = dXdt_full[2:end]
    nothing
end
