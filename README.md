# Rust-with-my-AI

Experimental and educational neural network library written from scratch in Rust.

This repository is aimed at learning and exploration, not production use. The core goal is to keep the mechanics visible: matrix ops, autograd, training loops, CNN blocks, and small Transformer-style components live in one place and can be read end to end.

## Status

- Experimental
- API may change without notice
- Good fit for study, prototyping, and small toy experiments
- Not positioned as a production ML framework

## What Is Included

- Dynamic autograd engine
- Basic matrix operations
- Dense layers and activations
- Optimizers: `SGD`, `Adam`
- CNN-oriented pieces such as `Conv2D`, `MaxPooling`, `Flatten`
- Transformer-oriented pieces such as `MultiHeadAttention`, positional encoding, residual blocks, layer norm, dropout
- A small character-level `TinyShakespeareGPT` example

## Quick Start

Prerequisite: a recent Rust toolchain with Cargo installed.

Run the test suite:

```bash
cargo test
```

Run the XOR example:

```bash
cargo run --example xor
```

Run specific model-oriented tests with output:

```bash
cargo test test_tiny_shakespeare_gpt -- --nocapture
cargo test test_mnist_cnn -- --nocapture
cargo test test_modern_components -- --nocapture
```

## Project Intent

This project is useful if you want to:

- inspect how backprop and parameter updates are wired together
- experiment with small neural building blocks in plain Rust
- learn by modifying a compact codebase instead of using a large framework

This project is not trying to compete with mature ecosystems such as PyTorch, JAX, or production inference stacks.

## Current Limitations

- The code favors readability and experimentation over raw performance
- Public API design is still fluid
- Model coverage is intentionally small
- Some examples and tests are toy-scale and intended as sanity checks, not benchmarks

## Roadmap

- Split the monolithic `src/lib.rs` into focused modules
- Add more deterministic tests around training behavior
- Improve documentation for tensor shapes and layer contracts
- Add a few cleaner end-to-end examples beyond XOR
- Reduce avoidable allocations and tighten hot paths

## Architecture Notes

The code follows a feature-first layout for several model components, especially around sequence-style operations. The implementation is intentionally explicit so the math path is easier to follow while debugging or studying.

## License

MIT. See [LICENSE](LICENSE).
