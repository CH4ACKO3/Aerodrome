"""An imported extension must not silently re-enable the interpreter lock."""
import sys
import sysconfig
import jax
import numpy
import scipy


def test_free_threaded_runtime_after_native_execution():
    assert sys.version_info[:2] == (3, 14), "use the pinned CPython 3.14t environment"
    assert sysconfig.get_config_var("Py_GIL_DISABLED") == 1
    jax.block_until_ready(jax.jit(lambda x: x*x)(jax.numpy.asarray(2.)))
    assert not sys._is_gil_enabled(), "a dependency or launch option enabled the GIL"
