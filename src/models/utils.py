def activation_for_substitute(xs, x):
    dead_idx = x == 0
    dead_idx.unsqueeze(-1).repeat(1, 1, 1, 1, xs.size(-1))
    xs[dead_idx] = 0
    return xs


def deploy(model):
    for name, module in model.named_modules():
        if hasattr(module, 're_parameterize'):
            module.re_parameterize()
    # if hasattr(model.fc, 're_parameterize'):
    #     model.fc.re_parameterize()
