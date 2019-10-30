module AdvancedTopics

import Distributions
import KernelDensity
import LinearAlgebra
import MLDatasets
import Plots
import Random

include("activation_functions.jl")
include("common.jl")
include("elm.jl")
include("sieve.jl")

include("mnist_data.jl")
include("synthetic_data.jl")
include("error.jl")

export ELM, Sieve, DataPassingSieve, ProjectionPassingSieve, predict,
       Cube, Linear, ReLU, Sigmoid, Square, Tanh,
       KDEComparator, train_kde_comparators,
       calculate_error,
       mnist_training_data, mnist_testing_data,
       synthetic_circular_data, synthetic_spiral_data

end # module
