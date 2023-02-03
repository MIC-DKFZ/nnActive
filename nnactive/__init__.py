from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """Error handled wrapper around importlib's version

    Returns:
        the installed version of nnactive
    """
    try:
        return version("nnactive")
    except PackageNotFoundError:
        # package is not installed
        import setuptools_scm

        return setuptools_scm.get_version(root="..", relative_to=__file__)

__version__ = get_version()
