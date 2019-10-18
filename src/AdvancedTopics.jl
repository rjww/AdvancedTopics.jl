module AdvancedTopics

import Distributions
import KernelDensity

include("kde_comparator.jl")
include("kde_sieve_layer.jl")
include("kde_sieve.jl")

export KDESieve, predict

end # module
