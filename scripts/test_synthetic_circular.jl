using AdvancedTopics

include("util.jl")

function perform_tests(training_samples::T₁,
                       training_targets::T₂,
                       testing_samples::T₃,
                       testing_targets::T₄,
                       n_neurons::Int;
                       boundary_offset::Int = 100) where {T₁ <: AbstractMatrix,
                                                          T₂ <: AbstractVector,
                                                          T₃ <: AbstractMatrix,
                                                          T₄ <: AbstractVector}
      X₀ = training_samples
      t₀ = training_targets
      X₁ = testing_samples
      t₁ = testing_targets
      L = n_neurons

      W, fs = train_kde_comparators(Float64, X₀, t₀, L, boundary_offset = boundary_offset)

      elm_sig = ELM{Float64}(X₀, t₀, W, [Sigmoid() for f in 1:length(fs)])
      elm_kde = ELM{Float64}(X₀, t₀, W, fs)

      y₀_sig = predict(elm_sig, X₀)
      y₀_kde = predict(elm_kde, X₀)
      y₁_sig = predict(elm_sig, X₁)
      y₁_kde = predict(elm_kde, X₁)

      plt_sig = plot_decision_boundary(elm_sig, X₀, t₀)
      plt_kde = plot_decision_boundary(elm_kde, X₀, t₀)

      training_error_sig = calculate_error(y₀_sig, t₀)
      training_error_kde = calculate_error(y₀_kde, t₀)
      testing_error_sig = calculate_error(y₁_sig, t₁)
      testing_error_kde = calculate_error(y₁_kde, t₁)

      (plt_sig, training_error_sig, testing_error_sig,
       plt_kde, training_error_kde, testing_error_kde)
end

for n_neurons in 1:2:5
    N = 200
    radius₁ = 4
    radius₂ = 11
    σ = 1.2

    X₀, t₀ = synthetic_circular_data(N, radius₁, radius₂, σ)
    X₁, t₁ = synthetic_circular_data(N, radius₁, radius₂, σ)

    (plt_sig, training_error_sig, testing_error_sig,
     plt_kde, training_error_kde, testing_error_kde) = perform_tests(X₀, t₀, X₁, t₁, n_neurons)

    savefig(plt_sig, "figures/db_circular_sig_$(n_neurons)_neurons.png")
    savefig(plt_kde, "figures/db_circular_kde_$(n_neurons)_neurons.png")
end
