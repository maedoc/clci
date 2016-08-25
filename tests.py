
from __future__ import absolute_import, print_function
import numpy as np
import numpy.testing as npt
import pyopencl as cl
import util
import model

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


@with_ctx_queue
def test_model(ctx, queue):
    class HMR(model.Model):
        "Hindmarsh-Rose model, cf http://www2.gsu.edu/~matals/hm_all_s.pdf"
        params = 'a b I c d e s x0'.split()
        auxs = [
            ('x2', 'x * x'),
        ]
        state_derivs = [
            ('x', 'y - a * x * x2 + b * x2 + I - z'),
            ('y', 'c - d * x2 - y'),
            ('z', 'e * (s * (x - x0) - z)'),
        ]
        consts = { 'a': 1.0, 'b': 3.0, 'c': -3.0, 'd': 5.0, 's': 4.0 }
    n = 16
    class HMR_Data(util.CLBase):
        state = util.Array((3, n))
        param = util.Array((3, n))
        cvars = util.Array((1, n)) # TODO
        deriv = util.Array((3, n))
    hmr = HMR()
    hmr.init_cl(ctx, queue)
    hmrd = HMR_Data()
    hmrd.init_cl(ctx, queue)
    hmrd.state[:] = np.random.rand(*hmrd.state.shape).astype('f')
    hmrd.param[:] = np.random.rand(*hmrd.param.shape).astype('f')
    x, y, z = hmrd.state.get()
    a, b, c, d, s = 1.0, 3.0, -3.0, 5.0, 4.0
    I, e, x0 = hmrd.param.get()
    x2 = x * x
    deriv = np.array([
        y - a * x * x2 + b * x2 + I - z,
        c - d * x2 - y,
        e * (s * (x - x0) - z),
    ])
    assert deriv.shape == hmrd.deriv.shape
    hmr.dfun(hmr.queue, (n, ), None, np.int32(n),
            hmrd.state.data, hmrd.param.data, hmrd.cvars.data,
            hmrd.deriv.data)
    npt.assert_allclose(deriv, hmrd.deriv.get(), 1e-5)
