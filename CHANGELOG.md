# Changelog

## 0.1.0

- Added a problem-agnostic adapter for additive parameterized linear systems.
- Added Latin-hypercube sampling, ordered solution bases, component-grid SVD
  compression, and versioned HDF5 storage with component-definition hashes.
- Added float64 JAX interpolation, batched reduced solves, gradients, and rank
  truncation, including unequal blocks and Hermitian or transpose projection.
- Added direct-solver gates, synchronized rank sweeps, and optional plotting.
- Added optional NumPyro NUTS with physical-coordinate prior-domain checks.
- Added an end-to-end jitr example, documentation and CPU CI.
- Importing DREAM enables JAX 64-bit mode globally.
