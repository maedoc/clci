
# Describe differential equation based model for OpenCL.
# See tests for example.

import numpy as np
import util


dfun_args = ['int stride']
dfun_args += ['__global float *%s' % s for s in 'state param input deriv'.split()]
dfun_args = ', '.join(dfun_args)
dfun_proto = '__kernel void dfun(%s)' % dfun_args
model_source_templ = """
{dfun_proto}
{{
    int id = get_global_id(0);

{params}

{states}

{auxs}

{derivatives}
}}
"""

param_templ = '    float %s = param[%d * stride + id];'
const_templ = '    #define %s %f'
aux_templ = '    float %s = %s;'
state_templ = '    float %s = state[%d * stride + id];'
deriv_templ = '    deriv[%d * stride + id] = %s;'

# TODO cvars

class Model(util.CLBase):
    """Base class for describing a differential equation model.

    After CL initialization, CL source corresponding to the
    below attributes is availabe with the `source` attribute,
    with the corresponding kernel as `dfun` attribute.

    Attributes
    ==========
    params: list of str
        names of parameters in model.
    consts: dict of str to float
        params taking constant values for model instance.
    auxs: dict of str to str
        names and expressions of auxiliary variables.
    state_derives: dict of str to str
        names of state variables and their time derivatives.

    """

    def init_cl(self, *args):
        # gen lines for param value defines or unpacking
        params = []
        consts = getattr(self, 'consts', {})
        i_param = 0 
        for name in self.params:
            if name in consts:
                params.append(const_templ % (name, consts[name]))
            else:
                params.append(param_templ % (name, i_param))
                i_param += 1
        # gen lines for auxiliaries
        auxs = []
        for (name, expr) in self.auxs:
            auxs.append(aux_templ % (name, expr))
        # gen lines for state unpack & derivatives
        states = []
        derivs = []
        for i, (state, deriv) in enumerate(self.state_derivs):
            states.append(state_templ % (state, i))
            derivs.append(deriv_templ % (i, deriv))
        # fill template from generated lines
        self.source = model_source_templ.format(
            dfun_proto=dfun_proto, 
            params='\n'.join(params),
            states='\n'.join(states),
            auxs='\n'.join(auxs),
            derivatives='\n'.join(derivs)
        )
        self.kernels = ['dfun']
        super(Model, self).init_cl(*args)
