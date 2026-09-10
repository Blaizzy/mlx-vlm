from ci.docs_check import Diagnostic, RepositorySnapshot, audit_snapshot


def snapshot(files):
    return RepositorySnapshot(frozenset(files), files)


def test_document_audit_accepts_existing_local_targets_and_nav():
    diagnostics = audit_snapshot(
        snapshot(
            {
                "README.md": "[Usage](docs/usage.md)",
                "docs/index.md": "![Image](assets/cat.jpg)",
                "docs/usage.md": "[Home](index.md#start)",
                "docs/assets/cat.jpg": "binary-placeholder",
                "mkdocs.yml": "docs_dir: docs\nnav:\n  - Home: index.md\n  - Usage: usage.md\n",
            }
        )
    )

    assert diagnostics == frozenset()


def test_document_audit_reports_missing_links_and_navigation():
    diagnostics = audit_snapshot(
        snapshot(
            {
                "docs/index.md": "[Missing](missing.md)",
                "mkdocs.yml": "docs_dir: docs\nnav:\n  - Gone: gone.md\n",
            }
        )
    )

    assert diagnostics == frozenset(
        {
            Diagnostic("docs/index.md", "missing_local_target", "missing.md"),
            Diagnostic("mkdocs.yml", "missing_nav_target", "gone.md"),
        }
    )


def test_document_audit_ignores_links_inside_code_blocks_and_external_links():
    diagnostics = audit_snapshot(
        snapshot(
            {
                "README.md": """
[Website](https://example.com)
`[Inline](missing.md)`
```markdown
[Example](also-missing.md)
```
""",
            }
        )
    )

    assert diagnostics == frozenset()
