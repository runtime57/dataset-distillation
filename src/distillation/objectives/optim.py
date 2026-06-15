def _sqrt_exact_forward_stable_backward(value, backward_eps=1e-12):
    exact = value.sqrt()
    stable = (value + backward_eps).sqrt()
    return exact.detach() + stable - stable.detach()


def adamw_step(
    params,
    gradients,
    first_moments,
    second_moments,
    step,
    lr,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.0,
    sqrt_backward_eps=1e-12,
):
    next_params = {}
    next_first_moments = {}
    next_second_moments = {}

    for (name, parameter), gradient in zip(params.items(), gradients):
        if gradient is None:
            next_params[name] = parameter
            next_first_moments[name] = first_moments[name]
            next_second_moments[name] = second_moments[name]
            continue

        first_moment = beta1 * first_moments[name] + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moments[name] + (1.0 - beta2) * gradient.square()
        first_unbiased = first_moment / (1.0 - beta1**step)
        second_unbiased = second_moment / (1.0 - beta2**step)
        decayed = parameter * (1.0 - lr * weight_decay)
        denominator = _sqrt_exact_forward_stable_backward(
            second_unbiased,
            backward_eps=sqrt_backward_eps,
        ).add(eps)
        next_params[name] = decayed - lr * first_unbiased / denominator
        next_first_moments[name] = first_moment
        next_second_moments[name] = second_moment

    return next_params, next_first_moments, next_second_moments
