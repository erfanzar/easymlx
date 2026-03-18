Contributing
============

Thank you for considering contributing to EasyMLX! This guide covers the
basics of setting up a development environment and submitting changes.

Development Setup
-----------------

1. Clone the repository::

    git clone https://github.com/erfanzar/easymlx.git
    cd easymlx

2. Create a virtual environment and install dependencies::

    uv sync

3. Install in editable mode with test dependencies::

    pip install -e ".[tests]"

Code Style
----------

EasyMLX uses:

- **Ruff** for linting and formatting (line length 121)
- **Black** for code formatting (line length 121, target Python 3.13)
- **basedpyright** for type checking

Run linting::

    ruff check easymlx/
    ruff format --check easymlx/

Running Tests
-------------

::

    pytest tests/

To skip slow tests::

    pytest tests/ -m "not slow"

Adding a New Model
------------------

See :doc:`infra/adding_models` for a step-by-step guide on adding a new
model architecture to EasyMLX.

Submitting Changes
------------------

1. Fork the repository and create a feature branch.
2. Make your changes with clear, descriptive commit messages.
3. Ensure all tests pass and code style checks are clean.
4. Open a pull request against the ``main`` branch.

License
-------

EasyMLX is licensed under the Apache License, Version 2.0. By contributing,
you agree that your contributions will be licensed under the same license.
