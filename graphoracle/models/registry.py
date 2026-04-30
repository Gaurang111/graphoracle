"""Model registry — maps string names to model classes."""

from __future__ import annotations

from typing import Any

from graphoracle.utils.exceptions import ModelNotRegisteredError


class ModelRegistry:
    """
    Global registry mapping model names to model classes.

    Usage
    -----
    @ModelRegistry.register("my_model")
    class MyModel(BaseForecastModel): ...

    # Later:
    model_cls = ModelRegistry.get("my_model")
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, model_cls: type | None = None) -> Any:
        """
        Register *model_cls* under *name*.

        Can be used as a plain call or as a decorator:

            ModelRegistry.register("foo", FooModel)
            @ModelRegistry.register("bar")
            class BarModel: ...
        """
        if model_cls is not None:
            cls._registry[name] = model_cls
            return model_cls

        def decorator(model_class: type) -> type:
            cls._registry[name] = model_class
            return model_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ModelNotRegisteredError(
                f"Model '{name}' not registered. Available: {available}"
            )
        return cls._registry[name]

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())
