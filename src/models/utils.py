def activation_for_substitute(xs, x):
    dead_idx = x == 0
    xs[dead_idx] = 0
    return xs


def deploy(model):
    for name, module in model.named_modules():
        if hasattr(module, 're_parameterize'):
            module.re_parameterize()