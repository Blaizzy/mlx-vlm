"""The CLI and the library must agree on every default they both define.

`mlx_vlm.generate` (the CLI) and `generate_step` (the library) used to declare
their own copies of the same constants, so one side could be changed without
the other. These tests compare the two by introspection rather than by
restating the values, so they keep working when a default is deliberately
changed and still fail when the two sides drift apart.
"""

import inspect
import sys
from unittest.mock import patch

from mlx_vlm.generate import ar, common, dispatch
from mlx_vlm.generate.ar import generate_step
from mlx_vlm.generate.dispatch import parse_arguments


def _cli_defaults():
    with patch.object(sys, "argv", ["mlx_vlm.generate"]):
        return vars(parse_arguments())


def _library_defaults():
    return {
        name: parameter.default
        for name, parameter in inspect.signature(generate_step).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def test_cli_and_library_agree_on_shared_defaults():
    cli, library = _cli_defaults(), _library_defaults()
    shared = sorted(set(cli) & set(library))

    # Guard against this test quietly comparing nothing: a rename on either
    # side would otherwise empty the intersection and still pass.
    assert len(shared) >= 10, f"expected many shared defaults, found {shared}"

    # A CLI default of None means "not given on the command line", and the
    # value is resolved further down (--draft-kind is inferred from the model
    # type, for instance). Any other disagreement is drift.
    mismatched = {
        name: {"cli": cli[name], "library": library[name]}
        for name in shared
        if cli[name] is not None and cli[name] != library[name]
    }
    assert not mismatched, (
        f"CLI and library defaults disagree: {mismatched}. Both sides should "
        "read the constant from mlx_vlm.generate.common."
    )


def test_shared_defaults_are_defined_once():
    """Modules that expose a default re-export `common`'s, never their own copy."""
    constants = [name for name in dir(common) if name.startswith("DEFAULT_")]
    assert constants, "no DEFAULT_* constants found in mlx_vlm.generate.common"

    for module in (ar, dispatch):
        for name in constants:
            if hasattr(module, name):
                assert getattr(module, name) is getattr(common, name), (
                    f"{module.__name__}.{name} is not the object defined in "
                    "mlx_vlm.generate.common"
                )
