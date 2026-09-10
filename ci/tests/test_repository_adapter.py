from ci.repository_adapter import main


def test_repository_adapter_exposes_the_shared_commands():
    for command in ("prepare", "plan", "hosted-checks", "report"):
        try:
            main([command, "--help"])
        except SystemExit as error:
            assert error.code == 0
