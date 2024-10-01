function min_indices(X)
    ind_vec = fill(0, length(X))
    n_ind = 0
    for i = 2:length(X) - 1
        if X[i - 1] > X[i] < X[i + 1]
            n_ind += 1
            ind_vec[n_ind] = i
        end
    end
    ind_vec = ind_vec[1:n_ind]
end

function amp(X)
    # being a continuous function, X = F(t) has to have interleaved min and max
    
    min_ind_vec = min_indices(X)
    max_ind_vec = min_indices(-X)
    n_max = min(length(max_ind_vec), length(min_ind_vec))    # the fewest min there can be
    amp_vec = X[max_ind_vec[1:n_max]] - X[min_ind_vec[1:n_max]]
end

# return the index at which convergence condition is reached
function is_converge(X; conv_len=10, tol=1e-4, burnin=0)
    
    i = burnin + 1    # force to discard burnin # steps
    
    while i + conv_len - 1 <= length(X)
        if all(abs.(X[i + 1:i + conv_len - 1] .- X[i]) .< tol)
            return i
        else
            i += conv_len
        end
    end
    
    return -1
end

function phospho_C(u)
    X = hcat(u...)    # 1-species, 2-time
    pC = transpose(1 .- X[[1], :]./sum(X, dims=1))
end

# return time_at_convergence, amplitude, period, phase
function attracted_to(f, X0, tmax0, p; tmax_stop=1e4, avg_window=32, amp_cutoff=1e-4, conv_tol=1e-4, burnin=0.0, callback=nothing)
    tmax = tmax0
    
    while tmax < tmax_stop
        prob = ODEProblem(f, copy(X0), (0.0, tmax), deepcopy(p))
        # Adaptive saving interval
        sol = solve(prob, AutoTsit5(Rosenbrock23()),
                    callback=callback, reltol=1e-8, abstol=1e-8, saveat=1e-4*tmax0)
        
        pC = phospho_C(sol.u)
        
        amp_vec = amp(pC)
        min_ind_vec = min_indices(pC)    # will be useful later
        
        burnin_steps = sum(min_ind_vec .<= sum(sol.t .< burnin))    # num of minima in burnin region
        conv_ind = is_converge(amp_vec, tol=conv_tol, burnin=burnin_steps)
        
        if conv_ind > 0
            window = min(avg_window, length(amp_vec) - conv_ind + 1)    # prevent ind out of bound
            
            amp_mean = sum(amp_vec[conv_ind:conv_ind + window - 1])/window
            if amp_mean < amp_cutoff
                return sol.t[conv_ind], -1, -1, -1    # stable fixed point
            end
            
            # otherwise, stable limit cycle
            t_window = sol.t[min_ind_vec[conv_ind:conv_ind + window - 1]]
            per_mean = sum(t_window[2:window] - t_window[1:window - 1])/(window - 1)
            phases = @. mod(-t_window, per_mean)/per_mean*2π    # phase at t = 0. trough defined as phase 0
            # println("phases = $(phases)")
            # TODO: phase variability within phases is big
            phase = sum(phases)/length(phases)
            return sol.t[min_ind_vec[conv_ind]], amp_mean, per_mean, phase
        else
            tmax *= 2
        end
    end
    
    # not converging. Often attracting to a fixed point/a limit cycle algebraically
    return -1, -1, -1, -1
end
