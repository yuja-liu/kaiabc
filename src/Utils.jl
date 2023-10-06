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
    n_max = length(max_ind_vec) - 1    # the fewest min there can be
    amp_vec = X[max_ind_vec[1:n_max]] - X[min_ind_vec[1:n_max]]
end

function is_converge(X; conv_len=10, tol=1e-4, burnin=0)
    
    i = burnin + 1    # force to discard burnin # steps
    
    while i + conv_len - 1 <= length(X)
        if all(abs.(X[i + 1:i + conv_len - 1] .- X[i]) .< tol)
            return true, i
        else
            i += conv_len
        end
    end
    
    return false, -1
end

function phospho_C(u)
    Xt = reduce(hcat, u)
    pC = 1 .- Xt[1, :]./reshape(sum(Xt, dims=1), :, 1)
end

function attracted_to(f, X0, tmax0, p; tmax_stop=1e4, avg_window=10, amp_cutoff=1e-4, conv_tol=1e-3, burnin=0.0, callback=nothing)
    converge = false
    tmax = tmax0
    
    while !converge && tmax < tmax_stop
        prob = ODEProblem(f, X0, (0.0, tmax), p)
        sol = solve(prob, callback=callback, reltol=1e-8, abstol=1e-8)
        
        pC = phospho_C(sol.u)
        
        amp_vec = amp(pC)
        min_ind_vec = min_indices(pC)    # will be useful later
        
        burnin_steps = sum(min_ind_vec .<= sum(sol.t .< burnin))    # num of minima in burnin region
        converge, conv_ind = is_converge(amp_vec, tol=conv_tol, burnin=burnin_steps)
        
        if converge
            window = min(avg_window, length(amp_vec) - conv_ind + 1)    # prevent ind out of bound
            
            amp_mean = sum(amp_vec[conv_ind:conv_ind + window - 1])/window
            if amp_mean < amp_cutoff
                return converge, -1, -1, -1    # stable fixed point
            end
            
            # otherwise, stable limit cycle
            t_window = sol.t[min_ind_vec[conv_ind:conv_ind + window - 1]]
            per_mean = sum(t_window[2:window] - t_window[1:window - 1])/(window - 1)
            phase = @. mod2pi(-t_window/per_mean*2π)    # phase at t = 0. trough defined as phase 0
            phase = sum(phase)/length(phase)
            return converge, amp_mean, per_mean, phase
        else
            tmax *= 2
        end
    end
    
    # not converging. Often attracting to a fixed point/a limit cycle algebraically
    return converge, -1, -1, -1
end
