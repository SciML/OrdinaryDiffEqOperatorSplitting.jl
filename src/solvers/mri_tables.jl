# ---------------------------------------------------------------------------
# Coefficient tables for IMEX-MRI-SR methods
#
# An IMEX-MRI-SR method is defined by `nΩ` slow-tendency coefficient matrices Ω^{k},
# a slow-implicit matrix Γ, and an abscissae vector c. This file holds the table type,
# the internal-consistency checks every table must satisfy, and the tables themselves.
# ---------------------------------------------------------------------------

"""
    MRISRCoefficients(Ω, Γ, c)

Coefficients of an `S`-stage IMEX-MRI-SR method, as defined in
[FisReyRob:2023:iem](@cite), Definitions 1.1 and 1.2.

`Ω` is a tuple of `nΩ` matrices, `Γ` a matrix, and `c` the slow abscissae. Following
the paper's notation, `Ω` and `Γ` carry the *embedding* as their last row, so each is
`(S+1) × S`:

```
Ω^{k} = [ 𝛀^{k} ; ω̂^{k} ],    Γ = [ 𝚪 ; γ̂ ]
```

The slow tendency coefficients are polynomials in the normalized fast time `τ`,
`ω_{i,j}(τ) = Σ_k ω_{i,j}^{k} τ^k`, so `Ω[k+1][i, j]` is the coefficient of `τ^k`.

The constructor checks the GARK internal-consistency conditions of Theorem 2.3, which
every IMEX-MRI-SR method must satisfy, along with the structural assumptions of
Definition 1.2. These are cheap and they are the only defence against a transcription
error in a table; the remaining order conditions are *not* checked, so a table that
constructs successfully can still fail to attain the order it claims.
"""
struct MRISRCoefficients{S, R, NΩ, T}
    Ω::NTuple{NΩ, NTuple{R, NTuple{S, T}}}   # Ω[k+1] is the τ^k matrix; row R = S+1 is ω̂^{k}
    Γ::NTuple{R, NTuple{S, T}}               # row R = S+1 is γ̂
    c::NTuple{S, T}
end

function MRISRCoefficients(
        Ω::Tuple{AbstractMatrix{T}, Vararg{AbstractMatrix{T}}},
        Γ::AbstractMatrix{T},
        c::AbstractVector{T},
    ) where {T}
    S = length(c)
    _validate_mrisr_table(Ω, Γ, c, S)

    Ωt = ntuple(k -> _rows_to_tuple(Ω[k], S), length(Ω))
    Γt = _rows_to_tuple(Γ, S)
    return MRISRCoefficients{S, S + 1, length(Ω), T}(Ωt, Γt, ntuple(i -> c[i], S))
end

_rows_to_tuple(M::AbstractMatrix, S::Int) =
    ntuple(i -> ntuple(j -> M[i, j], S), S + 1)

# The tables below are `vcat`s of comma-separated row vectors rather than matrix
# literals. Inside `[ ]` whitespace is significant and sign handling is context
# dependent: `[a -b]` is a two-element row while `[a - b]` is a one-element row holding a
# subtraction. Many entries here begin with a minus, so under literal syntax one space
# added or removed -- by a formatter, or by hand while aligning columns -- would silently
# fuse two coefficients into their difference, with no error and no shape change to
# catch it. Commas make that impossible.

function _validate_mrisr_table(Ω, Γ, c, S)
    nΩ = length(Ω)
    for (k, M) in enumerate(Ω)
        size(M) == (S + 1, S) || throw(
            ArgumentError(
                "Ω^{$(k - 1)} is $(size(M)) but must be $((S + 1, S)): one row per stage \
                 plus the embedding row, one column per stage."
            )
        )
    end
    size(Γ) == (S + 1, S) || throw(
        ArgumentError("Γ is $(size(Γ)) but must be $((S + 1, S)).")
    )

    # Definition 1.1 restarts every stage from y_n and forces the fast IVP over
    # [0, c_i H] with a 1/c_i factor, so a zero abscissa is only meaningful for the
    # first stage, which is Y_1 = y_n and takes no fast solve at all.
    iszero(c[1]) || throw(ArgumentError("c[1] is $(c[1]) but must be 0."))
    for i in 2:S
        iszero(c[i]) && throw(
            ArgumentError(
                "c[$i] is zero. Stage $i would force the fast IVP with a 1/c[$i] factor \
                 over an empty interval, which Definition 1.1 does not define."
            )
        )
    end

    # Definition 1.2: the first row of every coefficient matrix is identically zero
    # (stage 1 is y_n) and the Ω^{k} are strictly lower triangular.
    for (k, M) in enumerate(Ω)
        _require_zero_first_row(M, "Ω^{$(k - 1)}")
        for i in 1:S, j in i:S
            iszero(M[i, j]) || throw(
                ArgumentError(
                    "Ω^{$(k - 1)}[$i, $j] = $(M[i, j]) is on or above the diagonal, but \
                     Definition 1.2 assumes Ω^{$(k - 1)} is strictly lower triangular."
                )
            )
        end
    end
    _require_zero_first_row(Γ, "Γ")
    for i in 1:S, j in (i + 1):S
        iszero(Γ[i, j]) || throw(
            ArgumentError(
                "Γ[$i, $j] = $(Γ[i, j]) is above the diagonal, but Γ must be lower \
                 triangular; only its diagonal may couple a stage to itself."
            )
        )
    end

    # The embedding is evaluated over the whole step, so its abscissa is ĉ = 1 and its
    # last coefficient must vanish: ω̂_S = 0 because (1.5d) sums j ≤ S-1, and γ̂_S = 0
    # because that is what lets the embedded solution skip an implicit solve. Our
    # embedding pass relies on both, so require them rather than silently mis-stepping.
    for (k, M) in enumerate(Ω)
        iszero(M[S + 1, S]) || throw(
            ArgumentError(
                "ω̂^{$(k - 1)}[$S] = $(M[S + 1, S]) must be zero: (1.5d) sums the \
                 embedding's forcing over j ≤ $(S - 1) only."
            )
        )
    end
    iszero(Γ[S + 1, S]) || throw(
        ArgumentError(
            "γ̂[$S] = $(Γ[S + 1, S]) must be zero. A nonzero value would make the \
             embedded solution require its own implicit solve, which this \
             implementation does not perform."
        )
    )

    # Theorem 2.3, eq. (2.11a): Ω^{0} 𝟙 = c^{S}. The embedding row's abscissa is ĉ = 1.
    for i in 1:S
        rowsum = sum(Ω[1][i, j] for j in 1:S)
        rowsum == c[i] || throw(
            ArgumentError(
                "row $i of Ω^{0} sums to $rowsum but must equal c[$i] = $(c[i]) \
                 (internal consistency, eq. 2.11a)."
            )
        )
    end
    let rowsum = sum(Ω[1][S + 1, j] for j in 1:S)
        isone(rowsum) || throw(
            ArgumentError(
                "the embedding row of Ω^{0} sums to $rowsum but must equal ĉ = 1."
            )
        )
    end

    # Theorem 2.3, eq. (2.11b): Ω^{k} 𝟙 = 0 for k ≥ 1.
    for k in 2:nΩ, i in 1:(S + 1)
        rowsum = sum(Ω[k][i, j] for j in 1:S)
        iszero(rowsum) || throw(
            ArgumentError(
                "row $i of Ω^{$(k - 1)} sums to $rowsum but must be zero (internal \
                 consistency, eq. 2.11b)."
            )
        )
    end

    # Theorem 2.3, eq. (2.11c): Γ 𝟙 = 0.
    for i in 1:(S + 1)
        rowsum = sum(Γ[i, j] for j in 1:S)
        iszero(rowsum) || throw(
            ArgumentError(
                "row $i of Γ sums to $rowsum but must be zero (internal consistency, \
                 eq. 2.11c)."
            )
        )
    end
    return nothing
end

function _require_zero_first_row(M::AbstractMatrix, name)
    for j in axes(M, 2)
        iszero(M[1, j]) || throw(
            ArgumentError(
                "$name[1, $j] = $(M[1, j]) but the first row must be identically zero: \
                 stage 1 of an IMEX-MRI-SR method is Y_1 = y_n."
            )
        )
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

nstages(::MRISRCoefficients{S}) where {S} = S
num_omega(::MRISRCoefficients{S, R, NΩ}) where {S, R, NΩ} = NΩ

"""
    convert_coefficients(coeffs::MRISRCoefficients, ::Type{T})

Convert an exact (`Rational`) table to the working precision `T` of a state vector.
Done once per `init_cache` so the step itself does no rational arithmetic.
"""
function convert_coefficients(coeffs::MRISRCoefficients{S, R, NΩ}, ::Type{T}) where {S, R, NΩ, T}
    conv_rows(rows) = map(row -> map(T, row), rows)
    return MRISRCoefficients{S, R, NΩ, T}(
        map(conv_rows, coeffs.Ω),
        conv_rows(coeffs.Γ),
        map(T, coeffs.c),
    )
end

# ---------------------------------------------------------------------------
# IMEX-MRI-SR2(1) -- [FisReyRob:2023:iem](@cite), Appendix A
#
# Four stages, nΩ = 1, three slow nonlinear solves per step. Second order with a
# first-order embedding.
# ---------------------------------------------------------------------------

const IMEX_MRI_SR2_COEFFICIENTS = MRISRCoefficients(
    (
        # Ω^{0}; last row is ω̂^{0}
        vcat(
            [0, 0, 0, 0]',
            [3 // 5, 0, 0, 0]',
            [14 // 165, 2 // 11, 0, 0]',
            [-13 // 54, 137 // 270, 11 // 15, 0]',
            [-1 // 4, 1 // 2, 3 // 4, 0]',
        ),
    ),
    # Γ; last row is γ̂
    vcat(
        [0, 0, 0, 0]',
        [-11 // 23, 11 // 23, 0, 0]',
        [-6692 // 52371, -18355 // 52371, 11 // 23, 0]',
        [11621 // 90666, -215249 // 226665, 17287 // 50370, 11 // 23]',
        [-31 // 12, -1 // 6, 11 // 4, 0]',
    ),
    Rational{Int}[0, 3 // 5, 4 // 15, 1],
)

# ---------------------------------------------------------------------------
# IMEX-MRI-SR3(2) -- [FisReyRob:2023:iem](@cite), Appendix B
#
# Five stages, nΩ = 2, four slow nonlinear solves per step. Third order with a
# second-order embedding.
# ---------------------------------------------------------------------------

const IMEX_MRI_SR3_COEFFICIENTS = MRISRCoefficients(
    (
        # Ω^{0}; last row is ω̂^{0}
        vcat(
            [0, 0, 0, 0, 0]',
            [23 // 34, 0, 0, 0, 0]',
            [71 // 70, -3 // 14, 0, 0, 0]',
            [124 // 1155, 4 // 7, 5 // 11, 0, 0]',
            [162181 // 187680, 119 // 1380, 11 // 32, -5 // 17, 0]',
            [76355 // 74834, -46 // 31, 67 // 34, -36 // 71, 0]',
        ),
        # Ω^{1}; last row is ω̂^{1}.
        #
        # NOTE: the (4, 2) entry is *positive*, though Appendix B renders it with a
        # leading minus. A minus violates eq. (2.11b) -- over the common denominator
        # 1206582300 the row would sum to -19813883504 -- while a plus sums to exactly
        # zero: -2101267877 + 4·2476735438 - 575·13575085 = 0.
        vcat(
            [0, 0, 0, 0, 0]',
            [0, 0, 0, 0, 0]',
            [-14453 // 63825, 14453 // 63825, 0, 0, 0]',
            [-2101267877 // 1206582300, 2476735438 // 301645575, -13575085 // 2098404, 0, 0]',
            [
                -762580446799 // 588660102960, 11083240219 // 4328383110,
                -211274129 // 100368304, 89562055 // 106641323, 0,
            ]',
            [-3732974 // 2278035, 13857574 // 2278035, -52 // 9, 4 // 3, 0]',
        ),
    ),
    # Γ; last row is γ̂.
    #
    # NOTE: the (5, 2) and (5, 4) entries are *negative*, though Appendix B renders both
    # without a minus. As printed the row sums to 791340692/1009956059 rather than zero,
    # violating eq. (2.11c). Negating the two makes it sum to exactly zero and, checked
    # independently, makes the slow implicit base method A^{S,I} = Ω̄ + Γ satisfy both
    # third-order conditions on its last row exactly (Σb·c = 1/2 and Σb·c² = 1/3, with
    # b^{I,T} the last row by eq. 2.8c). No other pair of entries does.
    vcat(
        [0, 0, 0, 0, 0]',
        [-4 // 7, 4 // 7, 0, 0, 0]',
        [-2707004 // 3127425, 919904 // 3127425, 4 // 7, 0, 0]',
        [852879271 // 703839675, -1575000496 // 703839675, 5 // 11, 4 // 7, 0]',
        [
            43136869 // 2019912118, -73810600 // 1009956059,
            -17653551 // 87822266, -13993902 // 43911133, 4 // 7,
        ]',
        [-179 // 4140, 799 // 14490, 1 // 14, -1 // 12, 0]',
    ),
    Rational{Int}[0, 23 // 34, 4 // 5, 17 // 15, 1],
)
