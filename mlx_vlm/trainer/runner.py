"""Shared training and evaluation lifecycle.

This module will own the generic MLX optimization loop: accumulation,
evaluation cadence, checkpoint scheduling, callback invocation, and metrics.
It must remain modality-agnostic. Modality trainers pass their loss and
collation functions into this layer rather than duplicating the loop.
"""
