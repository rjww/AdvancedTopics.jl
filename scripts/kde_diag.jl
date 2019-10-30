using AdvancedTopics
using Distributions
using KernelDensity
using Plots

X = [rand(2, 100) -rand(2, 100)]
t = [ones(Int, 100); -ones(Int, 100)]
L = 1

W, fs = train_kde_comparators(Float64, X, t, L)

fig = plot(bg = :white)
for l in 1:L
    rng = range(-1.5, 1.5, length = 1000)
    plot!(rng, x -> pdf(fs[l].pdf₁, x), color = 1, xaxis = false, yaxis = false, legend = false, )
    plot!(rng, x -> pdf(fs[l].pdf₂, x), color = 2, xaxis = false, yaxis = false, legend = false)
end
fig

savefig(fig, "/home/rjww/Downloads/kdes.png")
