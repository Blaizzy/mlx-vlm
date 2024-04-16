import sys
from pathlib import Path

from setuptools import setup

package_dir = Path(__file__).parent / "mlx_vlm"
with open(package_dir / "requirements.txt") as fid:
    requirements = [l.strip() for l in fid.readlines()]

sys.path.append(str(package_dir))
from version import __version__

setup(
    name="mlx-vlm",
    version=__version__,
    description="Visual LLMs on Apple silicon with MLX and the Hugging Face Hub",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author_email="prince.gdt@gmail.com",
    author="MLX Contributors",
    url="https://github.com/Blaizzy/mlx-vlm",
    license="MIT",
    install_requires=requirements,
    packages=["mlx_vlm", "mlx_vlm.models"],
    python_requires=">=3.8",
)
