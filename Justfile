# NOTE: We automatically load a .env file containing the "GH_TOKEN" environment variable
# for use with semantic-release. If this isn't present, then those commands will likely fail.
set dotenv-load

# List all available recipes
default:
    @just --list

# Run the complete test suite
test:
    @echo "Running tests"
    uv run pytest

# Serve a local build of the project documentation at http://localhost:8000
serve-docs:
    @echo "Serving docs at http://localhost:8000"
    uv run --extra docs mkdocs serve

# Build the project documentation
build-docs:
    @echo "Building docs"
    uv run --extra docs mkdocs build

# Run the pre-commit hooks on all files in the repo
pre-commit:
    @echo "Running pre-commit hooks"
    uv run pre-commit run --all-files

# Run the coverage report
coverage:
    @echo "Running coverage report"
    uv run coverage run -m pytest
    uv run coverage report

# Determine the next version number
next-version:
    @echo "Determining next version"
    uv run semantic-release version --print

# Create a minor release
minor-release:
    @echo "Creating minor release"
    uv run semantic-release -vvv --noop version --minor --no-changelog

# Create a patch release
patch-release:
    @echo "Creating patch release"
    uv run semantic-release -vvv --noop version --patch --no-changelog

# Upload a release to PyPI
pypi-upload tag:
    @echo "Uploading release {{tag}} to PyPI"
    git checkout {{tag}}
    rm -rf dist
    uv run python -m build
    uv run twine upload dist/*
    git checkout -