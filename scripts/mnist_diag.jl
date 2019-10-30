using AdvancedTopics
using Images
using ImageTransformations

function generate_canvas(rows::Int, cols::Int)
    X, _ = mnist_training_data()
    N = last(size(X))
    imgs = reshape(X, (28, 28, N))
    side_length = 28

    canvas = Gray.(ones(side_length * (rows+1), side_length * (cols+1)))
    available = Set{Int}(1:N)

    for row in 2:rows, col in 2:cols
        r₀ = (row-1) * side_length + 1
        r₁ = r₀ + side_length - 1
        c₀ = (col-1) * side_length + 1
        c₁ = c₀ + side_length - 1

        n = rand(available)
        img = imgs[:,:,rand(available)]
        setdiff!(available, [n])

        canvas[r₀:r₁,c₀:c₁] .= (1 .- img)
    end

    canvas
end

canvas = generate_canvas(100, 20)
canvas = reverse(imrotate(canvas, π/2), dims = 2)
canvas = collect(canvas)
canvas = canvas[26:end-26,26:end-26]

save("/home/rjww/Downloads/mnist.png", canvas)
