# GraphOracle Documentation

## Install

```bash
git clone https://github.com/Gaurang111/graphoracle
cd graphoracle
pip install -e .
```

## Contents

- [Quickstart](quickstart.md)
- [Custom Models Guide](custom_models.md)
- [Custom Domains Guide](custom_domains.md)
- [API Reference](api_reference.md)

## What is GraphOracle?

GraphOracle is a framework for **multi-domain heterogeneous temporal graph forecasting**.
It wraps GNN architectures (HGT, TGN, STGNN, LSTM, GRU) with a unified training loop,
domain schema system, evaluation pipeline, and conformal uncertainty estimation.

The actual forecasting is performed by the underlying models. GraphOracle provides the
plumbing: data preparation, training, checkpointing, evaluation, and uncertainty wrapping.
It ships with 5 built-in domain schemas; new domains require a manual schema definition.

Built on PyTorch and PyTorch Geometric. Supports custom user-defined GNN models.
