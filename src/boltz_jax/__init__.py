from importlib.metadata import PackageNotFoundError, version

try:  # noqa: SIM105
    __version__ = version("boltz_jax")
except PackageNotFoundError:
    # package is not installed
    pass 