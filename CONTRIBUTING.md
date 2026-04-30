# Contributing to GraphOracle

Thank you for taking the time to contribute!

## Ways to Contribute

- **Bug reports** — open an issue with a minimal reproducible example
- **New built-in models** — subclass `BaseForecastModel`, register, open a PR
- **New domains** — subclass `BaseDomain`, add tests, open a PR
- **Benchmark results** — run on real datasets and share numbers
- **Documentation** — fix typos, add examples, improve API reference

---

## Development Setup

```bash
git clone https://github.com/Gaurang111/graphoracle
cd graphoracle

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install CPU-only PyTorch (faster for local dev)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric

# Install the package in editable mode with all dev extras
pip install -e ".[dev,viz,benchmarks]"

# Install pre-commit hooks
pre-commit install
```

---

## Running Tests

```bash
# Full suite
pytest tests/ -v

# Single file
pytest tests/test_custom_models.py -v

# With coverage report
pytest tests/ --cov=graphoracle --cov-report=html
# then open htmlcov/index.html
```

Test coverage must stay above **80%**. New features must include tests.

---

## Code Style

We use `ruff` for linting and `mypy` for type checking.

```bash
ruff check graphoracle/
mypy graphoracle/ --ignore-missing-imports
```

Rules:
- Type hints on all public functions
- Docstrings on all public classes and methods
- Max line length 100 characters
- No bare `except:` clauses

---

## Adding a New Model

1. Subclass `BaseForecastModel` in `graphoracle/models/your_model.py`
2. Implement `forward()` and `required_history_steps()`
3. Register: `ModelRegistry.register("your_model", YourModel)`
4. Import in `graphoracle/models/__init__.py`
5. Add tests in `tests/test_models.py`
6. Add a row to the models table in `README.md`

```python
# graphoracle/models/my_model.py
from graphoracle.models.base import BaseForecastModel
from graphoracle.models.registry import ModelRegistry

@ModelRegistry.register("my_model")
class MyModel(BaseForecastModel):
    def forward(self, graph, node_features, edge_index, temporal_encoding, memory=None):
        # node_features: {node_type: Tensor(N, T, F)}
        ...

    def required_history_steps(self) -> int:
        return 24
```

---

## Adding a New Domain

1. Subclass `BaseDomain` in `graphoracle/domains/your_domain.py`
2. Implement the `schema` property
3. Import in `graphoracle/domains/__init__.py`
4. Add tests in `tests/test_domains.py`

```python
from graphoracle.domains import BaseDomain
from graphoracle.graph import GraphSchema, NodeType, EdgeType

class MyDomain(BaseDomain):
    @property
    def schema(self) -> GraphSchema:
        return GraphSchema(
            node_types=[NodeType("my_node", features=["f1", "f2"], targets=["f1"])],
            edge_types=[EdgeType("my_edge", "my_node", "my_node")],
        )
```

---

## Pull Request Checklist

- [ ] Tests pass: `pytest tests/ -v`
- [ ] Coverage ≥ 80%: `pytest --cov=graphoracle --cov-fail-under=80`
- [ ] Linting passes: `ruff check graphoracle/`
- [ ] Type hints added on public API
- [ ] Relevant docs updated

---

## Reporting Bugs

Open an [issue](https://github.com/Gaurang111/graphoracle/issues) with:

1. **Description** — what you expected vs what happened
2. **Minimal reproducer** — the smallest code that triggers the bug
3. **Environment** — Python version, PyTorch version, OS
4. **Traceback** — full stack trace

---

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
