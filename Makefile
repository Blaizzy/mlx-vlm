# Makefile for virtualenv set and requirements management.
# See pip-tools documentation for maintenance documentation details.

.venv: requirements.txt
	python3 -m venv --clear .venv
	.venv/bin/pip install -r requirements.txt

requirements.txt: requirements.in .venv-deps
	.venv-deps/bin/pip-compile --generate-hashes requirements.in

# staging virtual environment for pip-tools setup
.venv-deps:
	python3 -m venv .venv-deps
	.venv-deps/bin/pip install pip-tools
