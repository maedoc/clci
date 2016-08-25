
from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
import util


def with_ctx_queue(f):
    "Decorator to pass build and pass context & queue."
    def _(*args, **kwds):
        ctx, queue = util.context_and_queue(util.create_cpu_context())
        return f(ctx, queue, *args, **kwds)
    return _


@with_ctx_queue
def test_cl_works(ctx, queue):
    "Smoke test that CL works, from PyOpenCL docs."
    a_np = np.random.rand(50000).astype(np.float32)
    b_np = np.random.rand(50000).astype(np.float32)
    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
    prg = cl.Program(ctx, """
    __kernel void sum(
        __global const float *a_g, __global const float *b_g, __global float *res_g)
    {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] + b_g[gid];
    }
    """).build()
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
    res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, res_g)
    assert (np.linalg.norm(res_np - (a_np + b_np))) == 0.0


def test_clbase():
    # make class which uses features of CLBase
    class Foo(util.CLBase):
        n = 16
        source = """
        #define i get_global_id(0)
        {prefix} add({args}) {{ z[i] = x[i] + y[i]; }}
        {prefix} sub({args}) {{ z[i] = x[i] - y[i]; }}
        """.format(prefix='__kernel void',
                args=', '.join(['__global float *%s' % s for s in 'xyz']))
        kernels = 'add sub'.split()
        x = util.Array((n, ), )
        y = util.Array(('m', ), ) # shape resolved at init_cl time
        z = util.Array(('n', ), )
        def __init__(self, m):
            self.m = m
    foo = Foo(Foo.n)
    ctx, queue = util.context_and_queue(util.create_cpu_context())
    foo.init_cl(ctx, queue)
    # check have attrs for kernels (and arrays)
    for name in 'add sub x y z'.split():
        assert hasattr(foo, name)
    # check arrays were built with correct type & shape
    for name in 'xyz':
        arr = getattr(foo, name)
        assert isinstance(arr, util.pyopencl.array.Array)
        assert arr.shape == (Foo.n, )
        assert arr.dtype == np.float32
    # run a kernel
    foo.x[:], foo.y[:] = x, y = np.random.rand(2, Foo.n).astype('f')
    foo.sub(foo.queue, (Foo.n, ), None, foo.x.data, foo.y.data, foo.z.data)
    # check result
    assert np.allclose(x - y, foo.z.get())
