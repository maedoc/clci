
import six
import os.path
import inspect
import pyopencl
import pyopencl.array


def create_cpu_context():
    "Return context for a CPU device type, or None if no CPU available."
    for platform in pyopencl.get_platforms():
        for device in platform.get_devices(pyopencl.device_type.CPU):
            return pyopencl.Context([device])


def context_and_queue(context):
    "Return tuple of given context and queue with that context."
    assert context
    return context, pyopencl.CommandQueue(context)


class Array(object):
    "Metadata for an ND array."

    def __init__(self, shape, dtype='f', **kwds):
        self.shape = shape
        self.dtype = dtype
        self.kwds = kwds


class CLBase(object):
    """Base class for handling OpenCL resources.

    Components which encapsulate OpenCL context, queue,
    programs, kernels and memory can be easily created by
    subclassed CLBase and providing special attributes listed
    below. Any attributes of type `Array` will result
    in the allocation of a `pyopencl.array.Array` instance.

    Attributes
    ==========
    context: pyopencl.CLContext
        A CL context with which CL resources are associated.
    queue: pyopencl.CommandQueue
        A CL queue in which transfers and invocations are placed for execution.
    source: str
        String of CL source code
    program: pyopencl.CLProgram
        CLProgram from which kernels are used, built from `source` if provided.
    kernels: list of str
        A list of kernel names to extract from the `program` attribute.

    """

    def effect_shape(self, name, shape):
        "Compute effective array shapes."
        # lookup rt dims, etc
        new_shape = []
        for dim in shape:
            if isinstance(dim, str):
                dim = int(getattr(self, dim))
            new_shape.append(dim)
        return tuple(new_shape)

    def init_cl(self, context, queue):
        "Initialize CL resources according to attributes in class decl."
        self.context = context
        self.queue = queue
        # build program from source
        if hasattr(self, 'source'):
            self.program = pyopencl.Program(
                    self.context, self.source).build()
        # pull out kernels
        for name in getattr(self, 'kernels', []):
            setattr(self, name, getattr(self.program, name))
        # setup data
        for name in dir(self):
            val = getattr(self, name)
            if isinstance(val, Array):
                shape = self.effect_shape(name, val.shape)
                array = pyopencl.array.Array(self.queue, shape, val.dtype)
                setattr(self, name, array)

    def set_arg(self, kernel, arg, value):
        getattr(self, kernel).set_arg(arg, value)

