"""Small, stateless helpers shared by trainer modules.

Use this module only for compact pure helpers such as padding calculations,
path normalization, and metric formatting. Stateful runtime code belongs in
``core.py``, ``runner.py``, or ``distributed.py`` instead.
"""

