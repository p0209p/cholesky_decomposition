include("./v1/cholesky_solve.jl")
include("./v2/cholesky_solve.jl")
include("./v3/cholesky_solve.jl")

using Plots, LinearAlgebra, LaTeXStrings, BenchmarkTools

import Random
Random.seed!(203)

function benchmark()
    t_default = []
    t_custom = [] # My implementation
    err = [] # Comparison of error between two implementations
    N = [5, 10, 20, 25, 50, 100, 200, 500, 1000, 2000]
    for s = N
        A = diagm(vec(rand(s, 1)))
        b = rand(s, 1)
        t = @elapsed xd = A \ b
        push!(t_default, t)
        t = @elapsed xc = cholesky_solve(A, b)
        push!(t_custom, t)
        push!(err, (sum((xd .- xc).^2) / s)^0.5)
    end
    plot(N, t_default, label="Julia's default implementation", marker=2, xlabel="Matrix size (n)", ylabel="Time (s)", title=L"x = A $\backslash$ b computation time")
    plot!(N, t_custom, marker=2, label="My implementation")
    savefig("./results/v1/time.png")

    plot(N, err, label="RMS Error", marker=2, xlabel="Matrix size (n)", ylabel="Error", title="Ax = b solution error")
    savefig("./results/v1/error.png")
end

function v2_benchmark()
    t_default = []
    t_custom = [] # My implementation
    err = [] # Comparison of error between two implementations
    N = [5, 10, 20, 25, 50, 100, 200, 500, 1000, 2000]
    for s = N
        A = diagm(vec(rand(s, 1)))
        b = rand(s, 1)
        t = @elapsed xd = A \ b
        push!(t_default, t)
        t = @elapsed xc = v2_cholesky_solve(A, b)
        push!(t_custom, t)
        push!(err, (sum((xd .- xc).^2) / s)^0.5)
    end
    plot(N, t_default, label="Julia's default implementation", marker=2, xlabel="Matrix size (n)", ylabel="Time (s)", title=L"x = A $\backslash$ b computation time")
    plot!(N, t_custom, marker=2, label="My implementation (v2)")
    savefig("./results/v2/time.png")

    plot(N, err, label="RMS Error", marker=2, xlabel="Matrix size (n)", ylabel="Error", title="Ax = b solution error")
    savefig("./results/v2/error.png")
end

function v3_benchmark()
    t_default = []
    t_custom = [] # My implementation
    err = [] # Comparison of error between two implementations
    N = [5, 10, 20, 25, 50, 100, 200, 500, 1000, 2000]
    for s = N
        A = diagm(vec(rand(s, 1)))
        b = rand(s, 1)
        t = @elapsed xd = A \ b
        push!(t_default, t)
        t = @elapsed xc = v3_cholesky_solve(A, b)
        push!(t_custom, t)
        push!(err, (sum((xd .- xc).^2) / s)^0.5)
    end
    plot(N, t_default, label="Julia's default implementation", marker=2, xlabel="Matrix size (n)", ylabel="Time (s)", title=L"x = A $\backslash$ b computation time")
    plot!(N, t_custom, marker=2, label="My implementation (v3)")
    savefig("./results/v3/time.png")

    plot(N, err, label="RMS Error", marker=2, xlabel="Matrix size (n)", ylabel="Error", title="Ax = b solution error")
    savefig("./results/v3/error.png")
end

function cholesky_benchmark()
    t_default = []
    t_custom = [] # My implementation
    N = [5, 10, 20, 25, 50, 100, 200, 500, 1000, 2000]
    for s = N
        A = diagm(vec(rand(s, 1)))
        t = @elapsed L = cholesky(A)
        push!(t_default, t)
        t = @elapsed L = my_cholesky(A)
        push!(t_custom, t)
    end
    plot(N, t_default, label="Julia's default implementation", marker=2, xlabel="Matrix size (n)", ylabel="Time (s)", title="Cholesky decomp. computation time")
    plot!(N, t_custom, marker=2, label="My implementation")
    savefig("./results/cholesky_decomp_comparison/cholesky_time.png")
end

cholesky_benchmark()
benchmark()
v2_benchmark()
v3_benchmark()
