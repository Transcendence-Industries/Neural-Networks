# Neural Networks in C

A small neural-network framework written in ANSI C using only standard libraries.

## Overview

This project demonstrates core deep-learning building blocks implemented from scratch: tensors, dense layers, activation/loss functions, forward + backward propagation, and mini-batch SGD.

## Highlights

- Multi-layer dense networks via a `Network` API
- Forward + backward propagation with cached intermediates
- Mini-batch SGD optimization
- Stable softmax + clipped cross-entropy
- Loss and accuracy APIs for classification and regression
- Versioned single-file model snapshots (`.nn`)
- Deterministic demo datasets and reproducible runs
- Unit + integration test coverage
- Address/Undefined Behavior sanitizer target

## Repository Layout

- `src/`: implementation files (`main.c`, network, layers, tensor, io, dataset, etc.)
- `include/`: public/project headers
- `tests/test_main.c`: test suite
- `data/`: shipped demo datasets
- `artifacts/`: generated model files (runtime output)

## Quick Start

```
make clean
make release
make run-demo
```

## Run Tests

```
make test
```

## Run Sanitizers

```
make san
```

## Demo Workflow

`src/main.c` includes two end-to-end pipelines:

1. Categorical classification
2. Regression

Each pipeline performs:

1. Load train/test CSV datasets
2. Train the network
3. Evaluate on test data
4. Save the model to `artifacts/*.nn`
5. Reload the model from disk
6. Verify prediction consistency after reload
