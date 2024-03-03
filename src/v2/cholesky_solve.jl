using LinearAlgebra

function v2_my_cholesky(A)
    r, c = size(A)
    if r != c
        throw(ArgumentError("Matrix must be square"))
    end
    L = zeros(r, c)
    L[1,1] = A[1,1]^0.5
    try
        for i = 2:r
            for j = 2:i
                if i == j
                    L[i,j] = (A[i,j] - sum(A[i,1:i-1].^2))^0.5
                else
                    L[i,j] = (A[i,j] - sum(A[i,1:j-1] .* A[j,1:j-1])) / L[j,j]
                end
            end
        end
        return L
    catch 
        throw(ArgumentError("A must be positive definite"))
    end
end

@inline function dot_(a, b)
    return sum(a .* b)
end

function v2_cholesky_solve(A, b)
    if size(A)[2] != size(b)[1] 
        throw(ArgumentError("Columns in A must be same as rows in b"))
    end
    r, c = size(A)
    """
    Ax = b

    Using cholesky decomposition, 
    LL* x = b, L is a lower triangular matrix, L* is conjugate transpose

    We have
    Ly = b, which can be solved by forward substitution
    
    Now, 
    L* x = y can be solved using back substitution
    """
    L = v2_my_cholesky(A) 
    Lstar = transpose(L)

    # Forward Substitution
    y = zeros(c, 1)
    y[1] = b[1] / L[1,1]
    for i = 2:c
        y[i] = (b[i] - dot_(L[i,1:i-1], y[1:i-1])) / L[i,i]
    end

    # Backward Substitution
    x = zeros(c, 1)
    x[c] = y[c] / Lstar[r,c]
    for i = c-1:-1:1
        x[i] = (y[i] - dot_(Lstar[i,c-1:i], x[c-1:i])) / Lstar[i,i]
    end

    return x
end