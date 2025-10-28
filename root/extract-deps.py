import argparse
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        print(
            "Error: 'tomli' is required for Python versions older than 3.11.\n"
            "Please install it using: pip install tomli"
        )
        sys.exit(1)


class PyprojectFormatError(Exception):
    """ Raised when the pyproject.toml file is malformed or missing required keys. """
    pass

def extract_dependencies(pyproject_path: Path | str) -> list[str]:
    """
    Parses a pyproject.toml file and extracts the project dependencies.

    Returns:
        list[str]: A list of dependencies in a pip-installable format.
    """
    try:
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        dependencies = pyproject_data["project"]["dependencies"]
        return dependencies
    except tomllib.TOMLDecodeError as e:
        raise PyprojectFormatError(f"The file '{ pyproject_path }' is not a valid TOML file.") from e
    except KeyError:
        raise PyprojectFormatError(
            f"The key '[project][dependencies]' was not found in '{ pyproject_path }'."
        ) from e

def main():
    """
    Main function to parse command-line arguments and print dependencies.
    """
    parser = argparse.ArgumentParser(
        description="Extract project.dependencies from a pyproject.toml file in a pip installable format."
    )
    parser.add_argument(
        "pyproject_file",
        nargs="?",
        default="pyproject.toml",
        help="Path to the pyproject.toml file (default: pyproject.toml in the current directory).",
    )
    args = parser.parse_args()

    try:
        dependencies = extract_dependencies(args.pyproject_file)
        for dep in dependencies:
            print(dep)
    except FileNotFoundError:
        print(f"Error: The file '{ args.pyproject_file }' was not found.", file=sys.stderr)
        sys.exit(1)
    except PyprojectFormatError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()