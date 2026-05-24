"""Lazy process exports.

This package originally imported every process module at import time, which
pulled in many optional training/media dependencies even for unrelated jobs.
For open-source installation smoke tests we only want to import the process
class that is actually requested by the config.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "BaseExtractProcess": ".BaseExtractProcess",
    "ExtractLoconProcess": ".ExtractLoconProcess",
    "ExtractLoraProcess": ".ExtractLoraProcess",
    "BaseProcess": ".BaseProcess",
    "BaseTrainProcess": ".BaseTrainProcess",
    "TrainVAEProcess": ".TrainVAEProcess",
    "BaseMergeProcess": ".BaseMergeProcess",
    "TrainSliderProcess": ".TrainSliderProcess",
    "TrainSliderProcessOld": ".TrainSliderProcessOld",
    "TrainSDRescaleProcess": ".TrainSDRescaleProcess",
    "ModRescaleLoraProcess": ".ModRescaleLoraProcess",
    "GenerateProcess": ".GenerateProcess",
    "BaseExtensionProcess": ".BaseExtensionProcess",
    "TrainESRGANProcess": ".TrainESRGANProcess",
    "BaseSDTrainProcess": ".BaseSDTrainProcess",
    "TrainFineTuneProcess": ".TrainFineTuneProcess",
    "MergeLoconProcess": ".MergeLoconProcess",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
