using LinearAlgebra

function v3_cholesky_solve(A, b)
    # if size(A)[2] != size(b)[1] 
    #     throw(ArgumentError("Columns in A must be same as rows in b"))
    # end
    """
    No decomposition decomposition, just forward substitution
    """
    # Forward Substitution
    return b .* (1 ./ diag(A))
end
