"""Host analysis frontends. Importing this module loads no plotting/control backend."""


def python_control():
    """Return upstream python-control, preserving its APIs and return conventions."""
    import control
    return control


def matlab_compat():
    """Return upstream control.matlab; partial MATLAB-like API, not MATLAB itself."""
    import control.matlab
    return control.matlab


def backend_status():
    """Installation status is not a claim that every upstream routine is tested."""
    from importlib.metadata import version,PackageNotFoundError
    packages = {}
    for name in ("control","scipy","slycot"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"packages":packages,"execution":"host CPU analysis; JAX only after explicit model conversion",
            "matlab_compatibility":"partial upstream API, no MATLAB engine attached"}
