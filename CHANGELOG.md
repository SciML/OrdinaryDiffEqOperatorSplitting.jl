# Changelog

## Unreleased

### Added

- `inner_dts` keyword argument to `init`, `solve`, and `reinit!` on
  `OperatorSplittingProblem`. Configures per-child sub-integrator step sizes
  for multirate operator splitting without having to mutate
  `integrator.child_subintegrators[i].dt` post-`init`. Backward compatible:
  omitting `inner_dts` broadcasts the outer `dt` to every child (the
  pre-existing single-rate behavior). See the usage docs for an example.
