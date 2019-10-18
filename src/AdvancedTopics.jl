module AdvancedTopics

import Distributions
import KernelDensity
import MLDatasets
import Random

include("kde_comparator.jl")
include("kde_sieve_layer.jl")
include("kde_sieve.jl")
include("mnist_data.jl")

export KDESieve, predict,
       training_data, testing_data

end # module
