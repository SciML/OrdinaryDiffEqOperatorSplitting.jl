# ---------------------------------------------------------------------------
# Tree addresses
# ---------------------------------------------------------------------------

"""
    SplitNode

Address of a node in an operator splitting tree, together with the object that lives
at that address.

Nodes are minted by indexing a [`GenericSplitFunction`](@ref) with integers. `f[]`
addresses the root, `f[i]` its `i`-th operator and `f[i, j]` the `j`-th operator of
that one:

```julia
node = f[2, 1]        # equivalently f[2][1]
node.path             # (2, 1)
node.object           # the addressed sub function
```

The splitting function defines the shape of the tree, so an address minted from it
denotes the same position in every tree that mirrors it and can be resolved against
each of them:

```julia
f[node]               # the sub function
alg[node]             # the inner algorithm
integrator[node]      # the sub integrator
```

Addresses are also how a [`TreeOption`](@ref) is told which node to configure.
"""
struct SplitNode{N, O}
    path::NTuple{N, Int}
    object::O
end

_path(is::Integer...) = map(Int, is)

# Human readable rendering of a path, used in error messages.
_showpath(::Tuple{}) = "the root operator"
_showpath(path::Tuple) = "operator [" * join(path, ", ") * "]"

"""
    _tree_child(obj, i)

The `i`-th child of `obj` in whichever splitting tree `obj` belongs to. Every tree
that mirrors a [`GenericSplitFunction`](@ref) implements this so that a
[`SplitNode`](@ref) can be resolved against it.
"""
function _tree_child(obj, i::Int)
    return throw(
        ArgumentError(
            "$(typeof(obj)) is a leaf of the splitting tree and has no operator $i to descend into."
        )
    )
end

function _tree_child(f::GenericSplitFunction, i::Int)
    checkbounds(Bool, 1:length(f.functions), i) || throw(
        ArgumentError(
            "operator $i is out of range: this splitting node has $(length(f.functions)) operators."
        )
    )
    return f.functions[i]
end

function _tree_child(alg::AbstractOperatorSplittingAlgorithm, i::Int)
    checkbounds(Bool, 1:length(alg.inner_algs), i) || throw(
        ArgumentError(
            "operator $i is out of range: $(nameof(typeof(alg))) has $(length(alg.inner_algs)) inner algorithms."
        )
    )
    return alg.inner_algs[i]
end

_resolve(obj, ::Tuple{}) = obj
_resolve(obj, path::Tuple) = _resolve(_tree_child(obj, first(path)), Base.tail(path))

_descend(node::SplitNode, ::Tuple{}) = node
function _descend(node::SplitNode, path::Tuple)
    i = first(path)
    child = _tree_child(node.object, i)
    return _descend(SplitNode((node.path..., i), child), Base.tail(path))
end

# --- minting addresses from the splitting function ---
Base.getindex(f::GenericSplitFunction) = SplitNode((), f)
Base.getindex(f::GenericSplitFunction, i::Integer, is::Integer...) =
    _descend(SplitNode((), f), _path(i, is...))
Base.getindex(node::SplitNode, i::Integer, is::Integer...) = _descend(node, _path(i, is...))

# --- resolving addresses against the mirroring trees ---
Base.getindex(f::GenericSplitFunction, node::SplitNode) = _resolve(f, node.path)
Base.getindex(alg::AbstractOperatorSplittingAlgorithm, node::SplitNode) =
    _resolve(alg, node.path)
Base.getindex(alg::AbstractOperatorSplittingAlgorithm, i::Integer, is::Integer...) =
    _resolve(alg, _path(i, is...))

get_operator(f::GenericSplitFunction, node::SplitNode) = _resolve(f, node.path)

function Base.show(io::IO, node::SplitNode)
    print(io, "f[", join(node.path, ", "), "]")
    return
end
function Base.show(io::IO, ::MIME"text/plain", node::SplitNode)
    show(io, node)
    print(io, " -> ", nameof(typeof(node.object)))
    return
end

# ---------------------------------------------------------------------------
# Per-node options
# ---------------------------------------------------------------------------

"""
    TreeOption(f::GenericSplitFunction, default)
    TreeOption{T}(f::GenericSplitFunction, default)

A value per node of an operator splitting tree, used to give individual
subintegrators their own settings where `init` only accepts one value for all of
them.

The option mirrors the shape of `f` and starts out holding `default` everywhere.
Plain assignment sets a single node; broadcast assignment sets a node and everything
below it:

```julia
dt = TreeOption(f, 1.0e-2)
dt[2]     = 1.0e-4          # this node alone
dt[2]    .= 1.0e-4          # this node and all operators below it
dt       .= 1.0e-4          # every node
dt[f[2]]  = 1.0e-4          # addressed by a SplitNode instead of a path
dt[2, 1]                    # read the value at [2, 1]
```

Later assignments overwrite earlier ones, so a broadcast over a subtree clobbers more
specific values written before it.

The element type is fixed by `default`; use `TreeOption{Any}(f, default)` for an
option whose values are of mixed type, such as a step size controller.
"""
mutable struct TreeOption{T}
    value::T
    const children::Vector{TreeOption{T}}
end

TreeOption(f::GenericSplitFunction, default::T) where {T} = _build_option(f, default, T)
TreeOption{T}(f::GenericSplitFunction, default) where {T} =
    _build_option(f, _checked_convert(T, default), T)

function _build_option(f::GenericSplitFunction, value::T, ::Type{T}) where {T}
    children = TreeOption{T}[_build_option(fi, value, T) for fi in f.functions]
    return TreeOption{T}(value, children)
end
_build_option(_, value::T, ::Type{T}) where {T} = TreeOption{T}(value, TreeOption{T}[])

function _checked_convert(::Type{T}, v) where {T}
    return try
        convert(T, v)
    catch err
        err isa MethodError || rethrow()
        throw(
            ArgumentError(
                "cannot store a value of type $(typeof(v)) in a TreeOption{$T}. \
                Construct the option as `TreeOption{Any}(f, default)` if it has to hold values of mixed type."
            )
        )
    end
end

_option_node(opt::TreeOption, ::Tuple{}) = opt
function _option_node(opt::TreeOption, path::Tuple)
    i = first(path)
    isempty(opt.children) && throw(
        ArgumentError(
            "$(_showpath(path)) does not exist: this option mirrors a leaf operator, which has no operators below it."
        )
    )
    checkbounds(Bool, opt.children, i) || throw(
        ArgumentError(
            "operator $i is out of range: this splitting node has $(length(opt.children)) operators."
        )
    )
    return _option_node(opt.children[i], Base.tail(path))
end

_store!(opt::TreeOption{T}, v) where {T} = (opt.value = _checked_convert(T, v))

function _fill_subtree!(opt::TreeOption, v)
    _store!(opt, v)
    for child in opt.children
        _fill_subtree!(child, v)
    end
    return opt
end

Base.getindex(opt::TreeOption) = opt.value
Base.getindex(opt::TreeOption, i::Integer, is::Integer...) =
    _option_node(opt, _path(i, is...)).value
Base.getindex(opt::TreeOption, node::SplitNode) = _option_node(opt, node.path).value

Base.setindex!(opt::TreeOption, v) = _store!(opt, v)
Base.setindex!(opt::TreeOption, v, i::Integer, is::Integer...) =
    _store!(_option_node(opt, _path(i, is...)), v)
Base.setindex!(opt::TreeOption, v, node::SplitNode) =
    _store!(_option_node(opt, node.path), v)

"""
    TreeOptionSubtree

The target of a broadcast assignment into a [`TreeOption`](@ref), produced by
`Base.dotview`. Assigning into it writes one value to a node and all of its
descendants.
"""
struct TreeOptionSubtree{T}
    node::TreeOption{T}
end

Base.dotview(opt::TreeOption) = TreeOptionSubtree(opt)
Base.dotview(opt::TreeOption, i::Integer, is::Integer...) =
    TreeOptionSubtree(_option_node(opt, _path(i, is...)))
Base.dotview(opt::TreeOption, node::SplitNode) =
    TreeOptionSubtree(_option_node(opt, node.path))

const _Broadcasted = Base.Broadcast.Broadcasted

# A TreeOption is not a collection as far as broadcasting is concerned. Without this
# the default `broadcastable` tries to `collect` it, and `opt .op= x` fails with a
# MethodError about iteration instead of the explanation in `_broadcast_value`.
Base.broadcastable(opt::TreeOption) = Ref(opt)

# Broadcast assignment into a TreeOption only ever means "give every node of this
# subtree the same value". Anything else that lowers to the same call is rejected
# rather than silently given a made up meaning -- see `_broadcast_value`.
Base.materialize!(dest::TreeOptionSubtree, bc::_Broadcasted) =
    _fill_subtree!(dest.node, _broadcast_value(bc))
Base.materialize!(dest::TreeOption, bc::_Broadcasted) =
    _fill_subtree!(dest, _broadcast_value(bc))

function _broadcast_value(bc::_Broadcasted)
    # `opt[i] .op= x` lowers to a broadcast of `op` whose first argument is the value
    # read back from the *single* node `opt[i]`, while the assignment target is the
    # whole subtree. There is no reading of that which is not surprising, so refuse.
    bc.f === identity || throw(
        ArgumentError(
            "`.$(bc.f)=` is not supported for a TreeOption: it reads the value of a single node \
            but writes to that node and everything below it. Write `opt[...] .= $(bc.f)(opt[...], x)` \
            with an undotted right hand side if that is what you intend."
        )
    )
    length(bc.args) == 1 || throw(
        ArgumentError("expected a single value to broadcast over the subtree, got $(length(bc.args)).")
    )
    v = only(bc.args)
    # Values that are not `broadcastable` arrive wrapped in a Ref.
    v isa Ref && return v[]
    return if v isa Union{AbstractArray, Tuple, _Broadcasted}
        throw(
            ArgumentError(
                "broadcast assignment into a TreeOption expects one scalar value, got $(typeof(v)). \
                Every node of the subtree receives the same value, so there is nothing to distribute."
            )
        )
    else
        v
    end
end

"""
    structure_matches(opt::TreeOption, f)

Whether `opt` was built for a splitting function of the same shape as `f`.
"""
structure_matches(opt::TreeOption, f::GenericSplitFunction) =
    length(opt.children) == length(f.functions) &&
    all(structure_matches(o, fi) for (o, fi) in zip(opt.children, f.functions))
structure_matches(opt::TreeOption, _) = isempty(opt.children)

function Base.show(io::IO, ::MIME"text/plain", opt::TreeOption{T}) where {T}
    println(io, "TreeOption{", T, "}:")
    return _show_option(io, opt, ())
end

function _show_option(io::IO, opt::TreeOption, path::Tuple)
    println(io, "  [", join(path, ", "), "] => ", repr(opt.value))
    for (i, child) in enumerate(opt.children)
        _show_option(io, child, (path..., i))
    end
    return
end

# ---------------------------------------------------------------------------
# Frozen configuration tree
# ---------------------------------------------------------------------------

"""
    ConfigTree

The settings of every node of a splitting tree, resolved once when the integrator is
built. `values` holds this node's settings and `children` mirrors the operators of
the corresponding [`GenericSplitFunction`](@ref).

This is what the tree construction in `src/integrator.jl` consumes: unlike a
[`TreeOption`](@ref), which is a mutable sparse thing the user edits, a `ConfigTree`
is immutable and concretely typed, so `config.children[i]` is as inferable as
`alg.inner_algs[i]` and no lookup reaches the stepping loop.
"""
struct ConfigTree{V <: NamedTuple, C <: Tuple}
    values::V
    children::C
end

_tree_child(config::ConfigTree, i::Int) = config.children[i]
Base.getindex(config::ConfigTree, node::SplitNode) = _resolve(config, node.path)

"""
    build_config_tree(f::GenericSplitFunction, options::NamedTuple)

Resolve `options` -- each entry either one value for the whole tree or a
[`TreeOption`](@ref) with a value per node -- into a [`ConfigTree`](@ref) mirroring `f`.
"""
function build_config_tree(f::GenericSplitFunction, options::NamedTuple)
    for (name, opt) in pairs(options)
        if opt isa TreeOption && !structure_matches(opt, f)
            throw(
                ArgumentError(
                    "the TreeOption passed as `$name` mirrors a splitting function of a different \
                    shape than the one this problem was built from."
                )
            )
        end
    end
    return _config_tree(f, options)
end

function _config_tree(f::GenericSplitFunction, options::NamedTuple)
    children = ntuple(
        i -> _config_tree(f.functions[i], map(o -> _option_child(o, i), options)),
        length(f.functions)
    )
    return ConfigTree(map(_option_value, options), children)
end
_config_tree(_, options::NamedTuple) = ConfigTree(map(_option_value, options), ())

_option_value(opt::TreeOption) = opt.value
_option_value(v) = v
_option_child(opt::TreeOption, i::Int) = opt.children[i]
_option_child(v, ::Int) = v

function validate_dt_tree(config::ConfigTree, path::Tuple = ())
    dt = config.values.dt
    dt > zero(dt) || error("dt must be positive, but $(_showpath(path)) was given $dt.")
    for (i, child) in enumerate(config.children)
        validate_dt_tree(child, (path..., i))
    end
    return
end

"""
    signed_dt_tree(config, tdir, tType)

Give every node's `dt` the direction of integration and the time type of the problem.
"""
signed_dt_tree(config::ConfigTree, tdir, ::Type{tType}) where {tType} = ConfigTree(
    merge(config.values, (; dt = tdir * convert(tType, config.values.dt))),
    map(child -> signed_dt_tree(child, tdir, tType), config.children)
)

"""
    default_adaptive_option(f, alg)

The default `adaptive` setting of `init`: a [`TreeOption`](@ref) in which every node
adapts exactly if its own algorithm is adaptive. Passing a scalar `adaptive` instead
configures the whole tree, including leaves whose algorithm cannot comply.
"""
function default_adaptive_option(
        f::GenericSplitFunction, alg::AbstractOperatorSplittingAlgorithm
    )
    opt = TreeOption(f, SciMLBase.isadaptive(alg))
    _default_adaptive!(opt, alg)
    return opt
end

function _default_adaptive!(opt::TreeOption, alg)
    opt.value = SciMLBase.isadaptive(alg)
    inner = alg isa AbstractOperatorSplittingAlgorithm ? alg.inner_algs : ()
    for (child, inner_alg) in zip(opt.children, inner)
        _default_adaptive!(child, inner_alg)
    end
    return
end

"""
    warn_non_adaptive(alg, config)

Warn about every splitting node that was asked to be adaptive although its algorithm
is not. Leaf algorithms are left to the inner integrator, which does its own checking.
"""
function warn_non_adaptive(
        alg::AbstractOperatorSplittingAlgorithm, config::ConfigTree, path::Tuple = ()
    )
    if config.values.adaptive && _is_verbose(config.values.verbose) &&
            !SciMLBase.isadaptive(alg)
        @warn "The algorithm $alg at $(_showpath(path)) is not adaptive."
    end
    for (i, child) in enumerate(config.children)
        warn_non_adaptive(alg.inner_algs[i], child, (path..., i))
    end
    return
end
warn_non_adaptive(alg, config::ConfigTree, path::Tuple = ()) = nothing

# Settings every node handles itself; anything else a user passes to `init` is an
# option of the inner integrators and travels down to the leaves untouched.
const NODE_OPTION_KEYS = (:dt, :adaptive, :verbose, :controller)
inner_values(values::NamedTuple) =
    NamedTuple{filter(!in(NODE_OPTION_KEYS), keys(values))}(values)

# ... of which a splitting node understands these.
const SPLIT_OPTION_KEYS = (
    :dtmin, :dtmax, :failfactor, :isoutofdomain,
    :abstol, :reltol, :internalnorm,
    :qmin, :qmax, :gamma, :qsteady_min, :qsteady_max,
)

function split_integrator_options(values::NamedTuple)
    inner = inner_values(values)
    known = filter(in(SPLIT_OPTION_KEYS), keys(inner))
    return IntegratorOptions(;
        verbose = values.verbose,
        adaptive = values.adaptive,
        NamedTuple{known}(inner)...
    )
end
