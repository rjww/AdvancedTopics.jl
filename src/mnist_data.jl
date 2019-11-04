const filtering_labels = collect(0:9)#[0, 1]

function mnist_training_data()
    X, T = MLDatasets.MNIST.traindata()
    reshape_data(filter_data(X, T)...)
end

function mnist_training_data(n_samples::Int)
    random_sample(n_samples, mnist_training_data()...)
end

function mnist_testing_data()
    X, T = MLDatasets.MNIST.testdata()
    reshape_data(filter_data(X, T)...)
end

function mnist_testing_data(n_samples::Int)
    random_sample(n_samples, mnist_testing_data()...)
end

function filter_data(samples::T1,
                     targets::T2) where {T1 <: AbstractArray,
                                         T2 <: AbstractVector}
    indices = findall(x -> x ∈ filtering_labels, targets)
    (samples[:,:,indices], targets[indices])
end

function reshape_data(samples::T1,
                      targets::T2) where {T1 <: AbstractArray,
                                          T2 <: AbstractVector}
    (reshape_samples(samples), reshape_targets(targets))
end

function reshape_samples(samples::T) where {T <: AbstractArray}
    rows, cols, N = size(samples)
    reshape(samples, (rows * cols, N))
end

function reshape_targets(targets::T) where {T <: AbstractVector}
    [x ∈ [0, 1, 2, 3, 4] ? 1 : -1 for x ∈ targets]
    # [x == 1 ? 1 : -1 for x ∈ targets]
end

function random_sample(n_samples::Int,
                       samples::T1,
                       targets::T2) where {T1 <: AbstractArray,
                                           T2 <: AbstractVector}
    @assert n_samples <= length(targets) "Blah."
    range = length(targets)
    indices = Random.randperm(range)[1:n_samples]
    (samples[:,indices], targets[indices])
end
