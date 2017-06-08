import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable, Function
from collections import defaultdict
import graphviz

"""
This is a rather distorted implementation of graph visualization in PyTorch.

This implementation is distorted because PyTorch's autograd is undergoing refactoring right now.
- neither func.next_functions nor func.previous_functions can be relied upon
- BatchNorm's C backend does not follow the python Function interface
- I'm not even sure whether to use var.creator or var.grad_fn (apparently the source tree and wheel builds use different
  interface now)

As a result, we are forced to manually trace the graph, using 2 redundant mechanisms:
- Function.__call__: this allows us to trace all Function creations. Function corresponds to Op in TF
- Module.forward_hook: this is needed because the above method doesn't work for BatchNorm, as the current C backend does
  not follow the Python Function interface. 

To do graph visualization, follow these steps:
1. register hooks on model: register_vis_hooks(model)
2. pass data through model: output = model(input)
3. remove hooks           : remove_vis_hooks()
4. perform visualization  : save_visualization(name, format='svg') # name is a string without extension

"""


old_function__call__ = Function.__call__

def register_creator(inputs, creator, output):
    """
    In the forward pass, our Function.__call__ and BatchNorm.forward_hook both call this method to register the creators

    inputs: list of input variables
    creator: one of
        - Function
        - BatchNorm module
    output: a single output variable
    """
    cid = id(creator)
    oid = id(output)
    if oid in vars: 
        return
    # connect creator to input
    for input in inputs:
        iid = id(input)
        func_trace[cid][iid] = input
        # register input
        vars[iid] = input
    # connect output to creator
    assert type(output) not in [tuple, list, dict]
    var_trace[oid][cid] = creator
    # register creator and output and all inputs
    vars[oid] = output
    funcs[cid] = creator

hooks = []

def register_vis_hooks(model):
    global var_trace, func_trace, vars, funcs
    remove_vis_hooks()
    var_trace  = defaultdict(lambda: {})     # map oid to {cid:creator}
    func_trace = defaultdict(lambda: {})     # map cid to {iid:input}
    vars  = {}                               # map vid to Variable/Parameter
    funcs = {}                               # map cid to Function/BatchNorm module
    hooks = []                               # contains the forward hooks, needed for hook removal

    def hook_func(module, inputs, output):
        assert 'BatchNorm' in mod.__class__.__name__        # batchnorms don't have shared superclass
        inputs = list(inputs)
        for p in [module.weight, module.bias]:
            if p is not None:
                inputs.append(p)
        register_creator(inputs, module, output)

    for mod in model.modules():
        if 'BatchNorm' in mod.__class__.__name__:           # batchnorms don't have shared superclass
            hook = mod.register_forward_hook(hook_func)
            hooks.append(hook)

    def new_function__call__(self, *args, **kwargs):
        inputs =  [a for a in args            if isinstance(a, Variable)]
        inputs += [a for a in kwargs.values() if isinstance(a, Variable)]
        output = old_function__call__(self, *args, **kwargs)
        register_creator(inputs, self, output)
        return output

    Function.__call__ = new_function__call__


def remove_vis_hooks():
    for hook in hooks:
        hook.remove()

    Function.__call__ = old_function__call__


def save_visualization(name, format='svg'):
    g = graphviz.Digraph(format=format)
    def sizestr(var):
        size = [int(i) for i in list(var.size())]
        return str(size)
    # add variable nodes
    for vid, var in vars.iteritems():
        if isinstance(var, nn.Parameter):
            g.node(str(vid), label=sizestr(var), shape='ellipse', style='filled', fillcolor='red')
        elif isinstance(var, Variable):
            g.node(str(vid), label=sizestr(var), shape='ellipse', style='filled', fillcolor='lightblue')
        else:
            assert False, var.__class__
    # add creator nodes
    for cid in func_trace:
        creator = funcs[cid]
        g.node(str(cid), label=str(creator.__class__.__name__), shape='rectangle', style='filled', fillcolor='orange')
    # add edges between creator and inputs
    for cid in func_trace:
        for iid in func_trace[cid]:
            g.edge(str(iid), str(cid))
    # add edges between outputs and creators
    for oid in var_trace:
        for cid in var_trace[oid]:
            g.edge(str(cid), str(oid))
    g.render(name)

