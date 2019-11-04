using AdvancedTopics
using DataFrames
using JLD2

include("util.jl")

function perform_tests(training_samples::T₁,
                       training_targets::T₂,
                       testing_samples::T₃,
                       testing_targets::T₄,
                       n_neurons::Int;
                       boundary_offset::Int = 1,
                       batch_size::Int = 100) where {T₁ <: AbstractMatrix,
                                                     T₂ <: AbstractVector,
                                                     T₃ <: AbstractMatrix,
                                                     T₄ <: AbstractVector}
    X₀ = training_samples
    t₀ = training_targets
    X₁ = testing_samples
    t₁ = testing_targets
    L = n_neurons

    W, fs = train_kde_comparators(Float64, X₀, t₀, L, boundary_offset = boundary_offset)

    elm_sig = ELM{Float64}(X₀, t₀, W, [Sigmoid() for f in 1:length(fs)], batch_size = batch_size)
    elm_kde = ELM{Float64}(X₀, t₀, W, fs, batch_size = batch_size)

    y₀_sig = predict(elm_sig, X₀)
    y₀_kde = predict(elm_kde, X₀)
    y₁_sig = predict(elm_sig, X₁)
    y₁_kde = predict(elm_kde, X₁)

    training_error_sig = calculate_error(y₀_sig, t₀)
    training_error_kde = calculate_error(y₀_kde, t₀)
    testing_error_sig = calculate_error(y₁_sig, t₁)
    testing_error_kde = calculate_error(y₁_kde, t₁)

    (training_error_sig, training_error_kde,
     testing_error_sig, testing_error_kde)
end

function run_experiments()
    X₀, t₀ = mnist_training_data()
    X₁, t₁ = mnist_testing_data()
    n_neurons = [2500]
    trials = 1:30

    for L in n_neurons
        results = DataFrame(:trial => collect(trials),
                            :sigmoid_training_error => zeros(Float64, length(trials)),
                            :kde_training_error => zeros(Float64, length(trials)),
                            :sigmoid_testing_error => zeros(Float64, length(trials)),
                            :kde_testing_error => zeros(Float64, length(trials)))

        for trial in trials
            (training_error_sig, training_error_kde,
             testing_error_sig, testing_error_kde) = perform_tests(X₀, t₀, X₁, t₁, L, batch_size = 100)

            results[trial,:sigmoid_training_error] = training_error_sig
            results[trial,:kde_training_error] = training_error_kde
            results[trial,:sigmoid_testing_error] = testing_error_sig
            results[trial,:kde_testing_error] = testing_error_kde

            println("$(L) neurons, trial $(trial)")
        end

        @save "results/mnist_$(L)_neurons.jld2" results
    end
end

run_experiments()

@load "results/mnist_2500_neurons.jld2" results
@show results[:,:sigmoid_training_error]
@show results[:,:kde_training_error]
