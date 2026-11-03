"""Pure reusable loss math.

Loss modules must not load datasets, initialize distributed state, or know a
particular model's forward signature. Modality trainers compose these functions
with model calls and task-specific metrics.
"""
