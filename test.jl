using Random
using Distributions
using DataFrames
using LinearAlgebra
using Statistics
using Optim
using StatsBase
using PyCall

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
# Instrument Construction and Estimation
# ============================================================================

println("Constructing instruments and performing OLS estimation...")

# Compute ln_within_share
grouped = groupby(product_data, [:market_ids, :satellite])
product_data.group_share = transform(grouped, :shares => sum => :group_share).group_share
product_data.ln_within_share = log.(product_data.shares ./ product_data.group_share)

# Create nest-specific ln_within_share
product_data.ln_within_share_sat = product_data.ln_within_share .* product_data.satellite
product_data.ln_within_share_wired = product_data.ln_within_share .* product_data.wired

# Create quadratic and interaction columns

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
# 2SLS Estimation
# ============================================================================

println("\nConstructing instrument matrix and performing 2SLS (IV) estimation...")

# Endogenous variable: prices
D = product_data.prices

# Exogenous regressors (included in structural equation and in instruments)
W = hcat(product_data.x, product_data.satellite, product_data.wired)

# Additional instruments constructed earlier (exclude any endogenous vars)
# Use competitor sums and 'other in nest' variables as instruments
Z_extra = hcat(product_data.sum_x_competitors,
               product_data.sum_w_competitors,
               product_data.x_other_in_nest,
               product_data.w_other_in_nest)

# Full instrument matrix: include exogenous regressors and the extra instruments
Z = hcat(W, Z_extra)

# Construct the full regressor matrix X = [D | W]
X_iv = hcat(D, W)

# Dimensions
n, k_iv = size(X_iv)
nz = size(Z, 2)

# Projection matrices and intermediate quantities
ZZ = Z' * Z
ZZ_inv = inv(ZZ)
Pz = Z * ZZ_inv * Z'

# 2SLS (equivalent via projected regressors)
X_hat = Pz * X_iv
beta_hat_iv = (X_hat' * X_hat) \ (X_hat' * product_data.logit_delta)

# Residuals
res_iv = product_data.logit_delta .- X_iv * beta_hat_iv

# Compute heteroskedasticity-robust (HC0) covariance for 2SLS
# Following standard formula: Var(beta) = (X'Pz X)^{-1} * (X'Z ZZ^{-1} Z' S Z ZZ^{-1} Z' X) * (X'Pz X)^{-1}
S = Diagonal(res_iv .^ 2)
X_Pz_X = X_iv' * Pz * X_iv
XZ = X_iv' * Z
middle = XZ * ZZ_inv * (Z' * S * Z) * ZZ_inv * XZ'
cov_beta_iv = inv(X_Pz_X) * middle * inv(X_Pz_X)
se_beta_iv = sqrt.(abs.(diag(cov_beta_iv)))

# t-stats and p-values
t_stats_iv = beta_hat_iv ./ se_beta_iv
p_values_iv = 2 .* (1 .- cdf.(Normal(), abs.(t_stats_iv)))

println("2SLS IV Regression: ln(s_jt/s_0t) ~ prices (endog) + x + satellite + wired")
println("-" ^ 70)
for i in 1:length(param_names)
    println("$(rpad(param_names[i], 12)): $(round(beta_hat_iv[i], digits=3)) (SE: $(round(se_beta_iv[i], digits=3)), t: $(round(t_stats_iv[i], digits=2)), p: $(round(p_values_iv[i], digits=3)))")
end

# First-stage diagnostics: regress D on Z to get partial F for excluded instruments
# Regress D on full Z
pi_hat = ZZ_inv * (Z' * D)
D_hat_full = Z * pi_hat
res_first_full = D .- D_hat_full
SSR_full = sum(res_first_full .^ 2)

# Regress D on W only (reduced set)
ZW = hcat(W)
ZW_ZW = ZW' * ZW
ZW_inv = inv(ZW_ZW)
pi_reduced = ZW_inv * (ZW' * D)
D_hat_reduced = ZW * pi_reduced
res_first_reduced = D .- D_hat_reduced
SSR_reduced = sum(res_first_reduced .^ 2)

q = size(Z_extra, 2) # number of excluded instruments
df_num = q
df_den = n - size(Z, 2)
F_stat = ((SSR_reduced - SSR_full) / df_num) / (SSR_full / df_den)

println("\nFirst-stage partial F-stat for excluded instruments: $(round(F_stat, digits=3))")

println("IV estimation completed!")

# ============================================================================
# BLP GMM Estimation Implementation (from blp.jl by Matteo Courthoud)
# ============================================================================

println("\nImplementing BLP GMM estimation...")

"""
Compute shares implied by deltas and shocks (adapted for current model)
"""
function implied_shares(Xt_::Matrix, pt_::Vector, ζt_::Matrix, δt_::Vector, δ0::Matrix, β::Vector, σ::Vector)::Vector
    """Compute shares implied by deltas and shocks"""
    J = length(δt_)
    n_consumers = size(ζt_, 2)
    x_t = Xt_[:, 1]  # x characteristics
    sat_t = Xt_[:, 2]  # satellite indicators
    wired_t = Xt_[:, 3]  # wired indicators
    # Random part: σ_sat * ν₁ * sat + σ_wired * ν₂ * wired
    random_part = zeros(J, n_consumers)
    for j in 1:J
        for i in 1:n_consumers
            random_part[j, i] = σ[1] * ζt_[1, i] * sat_t[j] + σ[2] * ζt_[2, i] * wired_t[j]
        end
    end
    # Mean utility: β_price * p + β_x * x + β_sat * sat + β_wired * wired + δ + random
    u = zeros(J+1, n_consumers)
    for j in 1:J
        mean_util = β[1] * pt_[j] + β[2] * x_t[j] + β[3] * sat_t[j] + β[4] * wired_t[j]
        for i in 1:n_consumers
            u[j, i] = δt_[j] + mean_util + random_part[j, i]
        end
    end
    u[J+1, :] = δ0[:]
    if any(isnan, u)
        return fill(NaN, J)
    end
    e = exp.(u)
    sum_e = sum(e, dims=1) .+ 1e-10
    q_matrix = similar(e[1:J, :])
    for i in 1:n_consumers
        q_matrix[:, i] = e[1:J, i] ./ sum_e[i]
    end
    q = mean(q_matrix, dims=2)
    return q[:]
end

"""
Solve the inner loop: compute delta, given the shares (contraction mapping adapted for current model)
"""
function inner_loop(qt_::Vector, Xt_::Matrix, pt_::Vector, ζt_::Matrix, β::Vector, σ::Vector)::Vector
    """Solve the inner loop: compute delta, given the shares"""
    δt_ = zeros(size(qt_))  # Better initial guess
    δ0 = zeros(1, size(ζt_, 2))
    dist = 1
    iteration = 0

    # Iterate until convergence
    while (dist > 1e-8 && iteration < 1000)
        q = implied_shares(Xt_, pt_, ζt_, δt_, δ0, β, σ)
        if any(isnan, q)
            return fill(NaN, length(qt_))
        end
        δt2_ = δt_ + 0.1 * (log.(qt_) - log.(max.(q, 1e-10)))
        dist = max(abs.(δt2_ - δt_)...)
        δt_ = δt2_
        iteration += 1
    end
    return δt_
end

"""
Compute deltas for all markets
"""
function compute_delta(q_::Vector, X_::Matrix, ζ_::Matrix, T::Vector)::Vector
    """Compute residuals"""
    δ_ = zeros(size(T))

    # Loop over each market
    for t in unique(T)
        qt_ = q_[T.==t]                             # Quantity in market t
        Xt_ = X_[T.==t,:]                           # Characteristics in mkt t
        δ_[T.==t] = inner_loop(qt_, Xt_, ζ_)        # Solve inner loop
    end
    return δ_
end

"""
Compute deltas for all markets (adapted for current model)
"""
function compute_delta_blp(q_::Vector, p_::Vector, x_::Vector, sat_::Vector, wired_::Vector, ζ_::Matrix, T::Vector, β::Vector, σ::Vector)::Vector
    """Compute residuals for BLP"""
    δ_ = zeros(size(T))

    # Loop over each market
    for t in unique(T)
        mask = T .== t
        qt_ = q_[mask]                             # Quantity in market t
        pt_ = p_[mask]                             # Prices in market t
        xt_ = x_[mask]                             # x in market t
        satt_ = sat_[mask]                         # satellite in market t
        wiredt_ = wired_[mask]                     # wired in market t
        Xt_ = hcat(xt_, satt_, wiredt_)            # [x, satellite, wired] for market t
        ζt_ = ζ_                                   # Same ζ for all markets (consumer draws)
        δ_[mask] = inner_loop(qt_, Xt_, pt_, ζt_, β, σ)   # Solve inner loop
    end
    return δ_
end

"""
Compute residual, given delta (IV) - corrected for proper GMM
"""
function compute_xi(X_::Matrix, IV_::Matrix, δ_::Vector)::Tuple
    """Compute residual, given delta (IV)"""
    # Standard IV estimation: β = (IV'*X)^(-1) * (IV'*δ)
    β_ = inv(IV_' * X_) * (IV_' * δ_)
    ξ_ = δ_ - X_ * β_
    return ξ_, β_
end

"""
Compute GMM objective function
"""
function GMM(θ::Vector)::Tuple
    """Compute GMM objective function"""
    β = θ[1:4]  # [β_price, β_x, β_sat, β_wired]
    σ = θ[5:6]  # [σ_sat, σ_wired]
    δ_ = compute_delta_blp(q_blp, product_data.prices, product_data.x, product_data.satellite, product_data.wired, ζ_blp, T_blp, β, σ)   # Compute deltas
    if any(isnan, δ_)
        return Inf, zeros(4), zeros(length(q_blp))
    end
    ξ_, β_ = compute_xi(X_blp, IV_blp, δ_)                   # Compute residuals
    # Efficient computation: gmm = ||Z' ξ||^2 / n^2
    Z_xi = Z_blp' * ξ_
    gmm = dot(Z_xi, Z_xi) / length(ξ_)^2
    return gmm, β_, δ_
end

# Prepare data for BLP estimation (adapted for current model with heterogeneity on sat/wired)
T_blp = Int.(product_data.market_ids .+ 1)  # Convert to 1-based indexing
# For current model: X includes homogeneous characteristics (prices, x, satellite, wired) 
X_blp = hcat(product_data.prices, product_data.x, product_data.satellite, product_data.wired)  # 4 columns: prices, x, satellite, wired
q_blp = product_data.shares
# Instruments: exogenous characteristics
IV_blp = hcat(product_data.x, product_data.w, product_data.satellite, product_data.wired)  # 4 columns
Z_blp = hcat(IV_blp, product_data.sum_x_competitors, product_data.sum_w_competitors,
             product_data.x_other_in_nest, product_data.w_other_in_nest)  # 8 columns for GMM weighting

# Draw consumer heterogeneity shocks (for sat and wired characteristics)
n_consumers = 1000
# Heterogeneity on sat and wired: ν₁ ~ N(β₂, σ_sat²), ν₂ ~ N(β₃, σ_wired²)
ζ_blp = Matrix(hcat(rand(Normal(β₂, σ_satellite), n_consumers), rand(Normal(β₃, σ_wired), n_consumers))')

println("Running BLP GMM estimation...")

# Initial guess: close to true values
initial_θ = [-2.0, 1.0, 4.0, 4.0, 1.0, 1.0]

# Debug: test GMM at initial
println("Debug: Testing GMM at initial θ")
gmm_val, β_est, δ_est = GMM(initial_θ)
println("Debug: GMM value: $(gmm_val)")

# Optimize GMM objective
result = optimize(θ -> GMM(θ)[1], initial_θ, NelderMead(), Optim.Options(iterations=200, show_trace=false))
θ_hat = Optim.minimizer(result)
gmm_min = Optim.minimum(result)

println("BLP GMM optimization completed. Objective: $(round(gmm_min, digits=6))")
println("Estimated parameters: $(round.(θ_hat, digits=3))")

# Extract estimates
β_blp_hat = θ_hat[1:4]
σ_blp_hat = θ_hat[5:6]

println("BLP GMM Results:")
println("-" ^ 50)
println("BLP GMM Regression: δ ~ prices + x + satellite + wired (GMM with random coefficients)")
println("-" ^ 70)
blp_param_names = ["prices", "x", "satellite", "wired", "σ_satellite", "σ_wired"]
for i in 1:6
    println("$(rpad(blp_param_names[i], 15)): $(round(θ_hat[i], digits=3))")
end
println("True parameters: prices=$(α), x=$(β₁), satellite=$(β₂), wired=$(β₃), σ_satellite=$(σ_satellite), σ_wired=$(σ_wired)")
