function synthetic_spiral_data(N, degrees, start, σ)
    s = deg2rad(start)
    N₀ = floor(Int, N/2)

    X = Matrix{Float64}(undef, 2, N)
    t = [ones(Int, N₀); -ones(Int, N-N₀)]

    for n in 1:N₀
        θ = s + sqrt(first(rand(1))) * deg2rad(degrees)
        X[:,n] = [-cos(θ) * θ + σ * first(rand(1)), sin(θ) * θ + σ * first(rand(1))]
    end

    for n in N₀+1:N
        θ = s + sqrt(first(rand(1))) * deg2rad(degrees)
        X[:,n] = [cos(θ) * θ + σ * first(rand(1)), -sin(θ) * θ + σ * first(rand(1))]
    end

    X, t
end

function synthetic_circular_data(N, radius₁, radius₂, σ)
    N₀ = floor(Int, N/2)

    X = Matrix{Float64}(undef, 2, N)
    t = [ones(Int, N₀); -ones(Int, N-N₀)]

    for n in 1:N₀
        θ = n / (2 * π)
        X[1,n] = radius₁ * cos(θ) + σ * randn()
        X[2,n] = radius₁ * sin(θ) + σ * randn()
    end

    for n in N₀+1:N
        θ = n / (2 * π)
        X[1,n] = radius₂ * cos(θ) + σ * randn()
        X[2,n] = radius₂ * sin(θ) + σ * randn()
    end

    X, t
end
