using Random
using Distributions
using DataFrames
using LinearAlgebra
using Statistics
using Optim
using StatsBase

# Set random seed
Random.seed!(1995)

# Model parameters
const T = 600  # number of markets
const J = 4    # number of products per market
const α = -2.0
const β₁ = 1.0
const β₂ = 4.0
const β₃ = 4.0
const σ_satellite = 1.0
const σ_wired = 1.0
const γ₀ = 0.5
const γ₁ = 0.25

println("Starting BLP data generation in Julia...")

# Product data structure
# Create DataFrame with market_ids, firm_ids, product_ids
market_ids = repeat(0:(T-1), inner=J)
firm_ids = repeat([1, 2, 3, 4], outer=T)
product_ids = repeat(0:(J-1), outer=T)

product_data = DataFrame(
    market_ids = market_ids,
    firm_ids = firm_ids,
    product_ids = product_ids
)

# Exogenous variables: x_jt and w_jt as absolute values of iid standard normal draws
n_obs = nrow(product_data)
product_data.x = abs.(randn(n_obs))
product_data.w = abs.(randn(n_obs))

# Indicators
product_data.satellite = (product_data.firm_ids .∈ [[1, 2]]) .|> Int
product_data.wired = (product_data.firm_ids .∈ [[3, 4]]) .|> Int

# Unobservables: ξ_jt and ω_jt with covariance matrix [[1, 0.25], [0.25, 1]]
Σ = [1.0 0.25; 0.25 1.0]
A = cholesky(Σ).L
z = randn(n_obs, 2)
unobs = z * A
product_data.ξ = unobs[:, 1]  # demand unobservable
product_data.ω = unobs[:, 2]  # cost unobservable

println("Generated $(nrow(product_data)) observations across $T markets")
println("x range: $(minimum(product_data.x)) to $(maximum(product_data.x))")
println("w range: $(minimum(product_data.w)) to $(maximum(product_data.w))")

# Correlation check
ξ_ω_corr = cor(product_data.ξ, product_data.ω)
println("ξ-ω correlation: $(round(ξ_ω_corr, digits=3)) (target: 0.25)")

sat_count = sum(product_data.satellite)
wired_count = sum(product_data.wired)
println("Satellite products: $sat_count, Wired products: $wired_count")

# Marginal costs
product_data.mc = exp.(γ₀ .+ γ₁ .* product_data.w .+ product_data.ω ./ 8)

println("MC range: $(round(minimum(product_data.mc), digits=3)) to $(round(maximum(product_data.mc), digits=3))")
println("MC mean: $(round(mean(product_data.mc), digits=3)), median: $(round(median(product_data.mc), digits=3))")

println("Data generation completed!")

# ============================================================================
# Market Shares and Derivatives Computation
# ============================================================================

"""
Compute shares, derivatives, and inside_shares_draws efficiently in one pass.
Returns: (shares, derivatives, inside_shares_draws)
"""
function market_shares_and_derivatives(prices, market_data, nu_draws)
    J = nrow(market_data)
    n_draws = size(nu_draws, 1)
    x = market_data.x
    ξ = market_data.ξ
    sat = market_data.satellite
    wired = market_data.wired

    # Compute utilities for each draw and product: shape (n_draws, J)
    utilities = zeros(n_draws, J)
    for j in 1:J
        utilities[:, j] = β₁ * x[j] .+ ξ[j] .+ nu_draws[:, 1] .* sat[j] .+ nu_draws[:, 2] .* wired[j] .+ α .* prices[j]
    end

    # Add outside option
    utilities = hcat(utilities, zeros(n_draws))
    exp_u = exp.(utilities .- maximum(utilities, dims=2))
    choice_probs = exp_u ./ sum(exp_u, dims=2)

    inside_shares_draws = choice_probs[:, 1:J]

    # Shares: average over draws
    shares = mean(inside_shares_draws, dims=1)[:]

    # Derivatives: compute analytically from choice probabilities
    derivatives = zeros(J, J)
    for j in 1:J
        for k in 1:J
            indicator = Float64(j == k)
            deriv_draws = α .* inside_shares_draws[:, j] .* (indicator .- inside_shares_draws[:, k])
            derivatives[j, k] = mean(deriv_draws)
        end
    end

    return shares, derivatives, inside_shares_draws
end

# Pre-draw simulation draws (to avoid jittering)
println("Pre-drawing simulation draws...")
n_draws = 10000
all_nu_draws = [
    rand(MvNormal([β₂, β₃], Diagonal([σ_satellite^2, σ_wired^2])), n_draws)
    for _ in 1:T
]

println("Simulation draws completed!")

# ============================================================================
# Convergence Testing
# ============================================================================

"""
Test derivative stability across different numbers of simulation draws.
"""
function test_convergence(prices, market_data, nu_draws_full, draw_counts, n_reps=100)
    n_available = size(nu_draws_full, 1)
    stds = Float64[]

    for n_draws in draw_counts
        deriv_list = Matrix{Float64}[]
        for rep in 1:n_reps
            # Randomly sample n_draws from the pre-drawn samples
            indices = sample(1:n_available, n_draws, replace=true)
            nu_draws = nu_draws_full[indices, :]
            _, derivs, _ = market_shares_and_derivatives(prices, market_data, nu_draws)
            push!(deriv_list, derivs)
        end
        # Compute standard deviation across replications
        deriv_array = stack(deriv_list)
        deriv_std = std(deriv_array, dims=3)[:, :, 1]
        push!(stds, mean(deriv_std))
    end
    return stds
end

# ============================================================================
# Price Equilibrium Solving
# ============================================================================

"""
Morrow-Skerlos algorithm for price equilibrium
"""
function solve_prices_morrow_skerlos(market_data, mc_market, nu_draws, max_iter=100, tol=1e-6)
    prices = copy(mc_market)
    for iteration in 1:max_iter
        # Efficiently compute shares, derivatives, and inside_shares_draws in one pass
        shares, derivatives, inside_shares_draws = market_shares_and_derivatives(prices, market_data, nu_draws)

        Lambda = Diagonal(α .* shares)
        Gamma = α .* (inside_shares_draws' * inside_shares_draws) / size(nu_draws, 1)
        diff = prices - mc_market
        zeta = inv(Lambda) * (Gamma' * diff - shares)
        prices_new = mc_market + zeta
        foc_residual = Lambda * (prices - mc_market - zeta)
        if maximum(abs.(foc_residual)) < tol
            return prices_new, iteration
        end
        prices = 0.5 .* prices + 0.5 .* prices_new
    end
    return prices, max_iter
end

# ============================================================================
# Solve for Equilibrium Prices
# ============================================================================

println("Solving for equilibrium prices...")

# Test at initial prices (p = MC) for market 0
market_0 = filter(row -> row.market_ids == 0, product_data)
prices_init = market_0.mc
println("Market 0 has $(nrow(market_0)) products")
println("prices_init length: $(length(prices_init))")
draw_counts = [50, 100, 200, 500, 1000, 2000, 5000]
stds = test_convergence(prices_init, market_0, all_nu_draws[1], draw_counts)

println("FOC: (p_jt - mc_jt) * ∂s_jt/∂p_jt + s_jt = 0")
println("Rearranged: p_jt - mc_jt = - (∂s_jt/∂p_jt)⁻¹ * s_jt")

# Solve using direct method - commented out for now
# equilibrium_prices_direct = Vector{Vector{Float64}}()
# success_flags_direct = Bool[]
# 
# for t in 1:T
#     market_data = filter(row -> row.market_ids == t-1, product_data)
#     mc_market = market_data.mc
#     nu_draws = all_nu_draws[t]
#     prices_direct, success = solve_prices_direct(market_data, mc_market, nu_draws)
#     push!(equilibrium_prices_direct, prices_direct)
#     push!(success_flags_direct, success)
# end
# 
# success_count = sum(success_flags_direct)
# println("Question 2(c)i completed:")
# println("Direct nonlinear solver: $success_count/$T markets solved successfully")
# println("Success rate: $(round(100 * success_count/T, digits=1))%")
# 
# all_prices_direct = vcat(equilibrium_prices_direct...)
# price_range_text = "Price range: $(round(minimum(all_prices_direct), digits=3)) to $(round(maximum(all_prices_direct), digits=3))"
# println(price_range_text)
# price_stats_text = "Price mean: $(round(mean(all_prices_direct), digits=3)), std: $(round(std(all_prices_direct), digits=3))"
# println(price_stats_text)

# Solve using Morrow-Skerlos method
equilibrium_prices_ms = Vector{Vector{Float64}}()
iterations_ms = Int[]

for t in 1:T
    market_data = filter(row -> row.market_ids == t-1, product_data)
    mc_market = market_data.mc
    nu_draws = all_nu_draws[t]

    prices_ms, iters = solve_prices_morrow_skerlos(market_data, mc_market, nu_draws)
    push!(equilibrium_prices_ms, prices_ms)
    push!(iterations_ms, iters)
end

all_prices_ms = vcat(equilibrium_prices_ms...)
println("Question 2(c)ii completed:")
println("Morrow-Skerlos method: $T markets solved")
println("Average iterations: $(round(mean(iterations_ms), digits=1))")
println("Max iterations: $(maximum(iterations_ms))")
println("Price range: $(round(minimum(all_prices_ms), digits=3)) to $(round(maximum(all_prices_ms), digits=3))")
println("Price mean: $(round(mean(all_prices_ms), digits=3)), std: $(round(std(all_prices_ms), digits=3))")

# Use Morrow-Skerlos prices
product_data.prices = vcat(equilibrium_prices_ms...)

# Compute market shares at equilibrium prices
println("Computing market shares at equilibrium prices...")
product_data.shares = zeros(nrow(product_data))

for t in 1:T
    market_data = filter(row -> row.market_ids == t-1, product_data)
    prices_market = market_data.prices
    nu_draws = all_nu_draws[t]
    
    shares, _, _ = market_shares_and_derivatives(prices_market, market_data, nu_draws)
    
    # Store shares back in the dataframe
    market_indices = findall(product_data.market_ids .== t-1)
    product_data.shares[market_indices] = shares
end

println("Market shares computation completed!")

# ============================================================================
# Instrument Construction and OLS Estimation
# ============================================================================

println("Constructing instruments and performing OLS estimation...")

# Compute within-nest shares for nested logit
product_data.nest_total_share = zeros(nrow(product_data))
for t in 0:T-1
    market_mask = product_data.market_ids .== t
    sat_mask = product_data.satellite .== 1
    wired_mask = product_data.wired .== 1
    
    # Satellite nest total share
    sat_indices = findall(market_mask .& sat_mask)
    if !isempty(sat_indices)
        sat_total = sum(product_data.shares[sat_indices])
        product_data.nest_total_share[sat_indices] .= sat_total
    end
    
    # Wired nest total share  
    wired_indices = findall(market_mask .& wired_mask)
    if !isempty(wired_indices)
        wired_total = sum(product_data.shares[wired_indices])
        product_data.nest_total_share[wired_indices] .= wired_total
    end
end

product_data.within_share = product_data.shares ./ product_data.nest_total_share

# Log within-nest shares (avoid log(0) by using small epsilon)
epsilon = 1e-10
product_data.ln_within_share_sat = zeros(nrow(product_data))
product_data.ln_within_share_wired = zeros(nrow(product_data))

for i in 1:nrow(product_data)
    if product_data.satellite[i] == 1
        product_data.ln_within_share_sat[i] = log(max(product_data.within_share[i], epsilon))
    end
    if product_data.wired[i] == 1
        product_data.ln_within_share_wired[i] = log(max(product_data.within_share[i], epsilon))
    end
end

# Create quadratic and interaction columns
product_data.x2 = product_data.x .^ 2
product_data.w2 = product_data.w .^ 2
product_data.x_w = product_data.x .* product_data.w

# Sum over competing goods in market t
product_data.sum_x_competitors = zeros(nrow(product_data))
product_data.sum_w_competitors = zeros(nrow(product_data))

for t in 0:T-1
    market_mask = product_data.market_ids .== t
    market_indices = findall(market_mask)
    
    x_sum = sum(product_data.x[market_indices])
    w_sum = sum(product_data.w[market_indices])
    
    for i in market_indices
        product_data.sum_x_competitors[i] = x_sum - product_data.x[i]
        product_data.sum_w_competitors[i] = w_sum - product_data.w[i]
    end
end

# Index of the other good in the same nest
product_data.x_other_in_nest = zeros(nrow(product_data))
product_data.w_other_in_nest = zeros(nrow(product_data))

for t in 0:T-1
    market_mask = product_data.market_ids .== t
    
    # Satellite nest
    sat_mask = market_mask .& (product_data.satellite .== 1)
    sat_indices = findall(sat_mask)
    if length(sat_indices) == 2
        # Two satellite products
        product_data.x_other_in_nest[sat_indices[1]] = product_data.x[sat_indices[2]]
        product_data.x_other_in_nest[sat_indices[2]] = product_data.x[sat_indices[1]]
        product_data.w_other_in_nest[sat_indices[1]] = product_data.w[sat_indices[2]]
        product_data.w_other_in_nest[sat_indices[2]] = product_data.w[sat_indices[1]]
    end
    
    # Wired nest
    wired_mask = market_mask .& (product_data.wired .== 1)
    wired_indices = findall(wired_mask)
    if length(wired_indices) == 2
        # Two wired products
        product_data.x_other_in_nest[wired_indices[1]] = product_data.x[wired_indices[2]]
        product_data.x_other_in_nest[wired_indices[2]] = product_data.x[wired_indices[1]]
        product_data.w_other_in_nest[wired_indices[1]] = product_data.w[wired_indices[2]]
        product_data.w_other_in_nest[wired_indices[2]] = product_data.w[wired_indices[1]]
    end
end

println("Instrument construction completed!")

# ============================================================================
# OLS Estimation
# ============================================================================

# Compute outside shares for each market
product_data.outside_share = zeros(nrow(product_data))
for t in 0:T-1
    market_mask = product_data.market_ids .== t
    market_indices = findall(market_mask)
    total_share = sum(product_data.shares[market_indices])
    product_data.outside_share[market_indices] .= 1 .- total_share
end

# Compute logit delta: ln(s_jt / s_0t)
product_data.logit_delta = log.(product_data.shares ./ product_data.outside_share)

# OLS using matrix algebra (no intercept)
y = product_data.logit_delta
X = hcat(product_data.prices, product_data.x, product_data.satellite, product_data.wired)

# Compute OLS estimates: beta_hat = (X^T X)^(-1) X^T y
beta_hat_ols = (X' * X) \ (X' * y)

# Compute residuals and HC0 robust standard errors
y_hat = X * beta_hat_ols
residuals_ols = y - y_hat
n, k = size(X)

# HC0 robust covariance matrix
V = X' * Diagonal(residuals_ols .^ 2) * X
cov_matrix_ols = inv(X' * X) * V * inv(X' * X)
se_ols = sqrt.(diag(cov_matrix_ols))

# t-statistics and p-values
t_stats_ols = beta_hat_ols ./ se_ols
p_values_ols = 2 .* (1 .- cdf.(Normal(), abs.(t_stats_ols)))

println("OLS Regression: ln(s_jt/s_0t) ~ x + satellite + wired + prices (no intercept)")
println("-" ^ 70)
param_names = ["prices", "x", "satellite", "wired"]
for i in 1:length(param_names)
    println("$(rpad(param_names[i], 12)): $(round(beta_hat_ols[i], digits=3)) (SE: $(round(se_ols[i], digits=3)), t: $(round(t_stats_ols[i], digits=2)), p: $(round(p_values_ols[i], digits=3)))")
end

println("OLS estimation completed!")

# ============================================================================
# True Elasticities and Diversion Ratios Computation
# ============================================================================

"""
Compute RC elasticities using true parameters on OBSERVED shares.
This matches the Python implementation for fair comparison with estimated elasticities.
"""
function compute_rc_elasticities_observed_shares(market_df, nu_draws, alpha, beta_x, beta_sat, beta_wired, sigma_sat, sigma_wired)
    J = nrow(market_df)
    prices = market_df.prices
    observed_shares = market_df.shares
    x, satellite, wired = market_df.x, market_df.satellite, market_df.wired

    # Compute random coefficient deviations (the part that varies across individuals)
    # rc_deviation should be shape (n_draws, J)
    rc_deviation = sigma_sat * (nu_draws[:, 1] * satellite') + sigma_wired * (nu_draws[:, 2] * wired')

    # Back out mean utilities (delta) via contraction mapping
    # Goal: Find delta such that observed_shares = E[exp(delta + rc_deviation) / (1 + sum exp(delta + rc_deviation))]
    delta = log.(observed_shares)  # Initial guess

    for iteration in 1:1000
        # Compute individual choice probabilities
        utilities = delta' .+ rc_deviation  # Shape: (n_draws, J)
        exp_utils = exp.(utilities)
        choice_probs = exp_utils ./ (1 .+ sum(exp_utils, dims=2))  # Shape: (n_draws, J)

        # Predicted shares
        predicted_shares = mean(choice_probs, dims=1)[:]

        # Contraction update: delta_new = delta + log(s_obs) - log(s_pred)
        delta_new = delta .+ log.(observed_shares) .- log.(predicted_shares)

        # Check convergence
        if maximum(abs.(delta_new .- delta)) < 1e-14
            delta = delta_new
            break
        end
        delta = delta_new
    end

    # Compute final choice probabilities with converged delta
    utilities = delta' .+ rc_deviation
    exp_utils = exp.(utilities)
    choice_probs = exp_utils ./ (1 .+ sum(exp_utils, dims=2))

    # Compute elasticities using analytical derivatives
    elasticities = zeros(J, J)
    for j in 1:J
        for k in 1:J
            if j == k
                # Own-price: E[s_ij * (1 - s_ij)]
                deriv = alpha * mean(choice_probs[:, j] .* (1 .- choice_probs[:, j]))
            else
                # Cross-price: -E[s_ij * s_ik]
                deriv = -alpha * mean(choice_probs[:, j] .* choice_probs[:, k])
            end

            if observed_shares[j] > 1e-10
                elasticities[j, k] = (prices[k] / observed_shares[j]) * deriv
            end
        end
    end

    return elasticities
end

"""
Compute diversion ratios using PyBLP's derivative-based method.
This converts elasticities to Jacobian and replaces diagonal with outside option derivative.
"""
function compute_diversion_ratios_pyblp(elasticity_matrices, product_data, T, J)
    diversion_matrices = []

    for t in 1:T
        elast_matrix = elasticity_matrices[t]
        market_data_t = filter(row -> row.market_ids == t-1, product_data)
        shares = market_data_t.shares
        prices = market_data_t.prices

        # Convert elasticities to Jacobian (derivatives): ∂s_j/∂p_k = (s_j/p_k) * ε_jk
        jacobian = zeros(J, J)
        for j in 1:J
            for k in 1:J
                jacobian[j, k] = (shares[j] / prices[k]) * elast_matrix[j, k]
            end
        end

        # PyBLP's method: Replace diagonal with outside option derivative
        # ∂s_0/∂p_j = -Σ_k ∂s_k/∂p_j (by adding-up constraint)
        jacobian_diag = diag(jacobian)
        for j in 1:J
            jacobian[j, j] = -sum(jacobian[j, :])
        end

        # Compute diversion ratios: D_jk = -Jacobian[j,k] / Jacobian[j,j]
        diversion = -jacobian ./ jacobian_diag

        push!(diversion_matrices, diversion)
    end

    return diversion_matrices
end

println("Computing TRUE RC elasticities (true params, observed shares)...")
true_elasticity_matrices = [compute_rc_elasticities_observed_shares(
    filter(row -> row.market_ids == t-1, product_data), all_nu_draws[t], -2.0, 1.0, 4.0, 4.0, 1.0, 1.0
) for t in 1:T]

true_avg_elasticity = dropdims(mean(reshape(hcat(true_elasticity_matrices...), (J, J, T)), dims=3), dims=3)
own_elasticities_true = diag(true_avg_elasticity)

println("Computing TRUE RC diversion ratios...")
true_diversion_matrices = compute_diversion_ratios_pyblp(true_elasticity_matrices, product_data, T, J)
true_avg_diversion = dropdims(mean(reshape(hcat(true_diversion_matrices...), (J, J, T)), dims=3), dims=3)

# ============================================================================
# Display Results
# ============================================================================

println("\n" * "="^70)
println("TRUE ELASTICITIES AND DIVERSION RATIOS")
println("="^70)

println("\nAverage Own-Price Elasticities:")
product_labels = ["Satellite 1", "Satellite 2", "Wired 1", "Wired 2"]
for i in 1:J
    println("  $(product_labels[i]): $(round(own_elasticities_true[i], digits=3))")
end

println("\nTrue Diversion Ratios (RC Logit with TRUE params on OBSERVED shares):")
println("Diagonal = diversion to outside option D_j0")
for i in 1:J
    row_str = "  $(product_labels[i]): "
    for j in 1:J
        row_str *= "$(round(true_avg_diversion[i,j], digits=3)) "
    end
    println(row_str)
end

println("\nNote: True values computed using true parameters (-2, 1, 4, 4, 1, 1) on observed shares")