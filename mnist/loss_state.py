"""Adapted from VeLO code: https://github.com/google/learned_optimization/blob/687e72e7b5596dfb80c5196fd51f43058899edd9/learned_optimization/research/general_lopt/hyper_v2.py"""

import torch

import torch

class MomAccumulator:
    def __init__(self, m, t):
        self.m = m
        self.t = t

class RMSAccumulator:
    def __init__(self, rms, t):
        self.rms = rms
        self.t = t

def rolling_mom(decay: float):
    def init_fn(p):
        return MomAccumulator(m=torch.zeros_like(p), t=torch.tensor(0, dtype=torch.int32))

    def update_fn(state, grad):
        m = state.m * decay + (1 - decay) * grad
        return MomAccumulator(m=m, t=state.t + 1)

    return init_fn, update_fn

def rolling_rms(decay: float):
    def init_fn(p):
        return RMSAccumulator(rms=torch.zeros_like(p), t=torch.tensor(0, dtype=torch.int32))

    def update_fn(state, grad):
        clip_decay = torch.clip(decay, 0.0, 1.0)
        rms = state.rms * clip_decay + (1 - clip_decay) * (grad * grad)
        return RMSAccumulator(rms=rms, t=state.t + 1)

    return init_fn, update_fn

def _vmap_accumulator(accumulator, decays):
    def init_fn(p):
        return torch.stack([accumulator(d).init(p) for d in decays])

    def update(state, grads):
        return torch.stack([accumulator(d).update(s, g) for d, s, g in zip(decays, state, grads)])

    return init_fn, update

def vec_rolling_mom(decays):
    return _vmap_accumulator(rolling_mom, decays)

def vec_rolling_rms(decays):
    return _vmap_accumulator(rolling_rms, decays)

def safe_rsqrt(x):
    return torch.rsqrt(torch.max(x, torch.tensor(1e-9)))


def _fractional_tanh_embed(x):

    def one_freq(timescale):
        return torch.tanh((x - torch.tensor(float(timescale))) * 10)

    timescales = torch.tensor([0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1], dtype=torch.float32)
    return torch.jit.vectorize(one_freq)(timescales)


def factored_dims(shape):
    """Whether to use a factored second moment estimator.

    If there are not two dimensions of size >= min_dim_size_to_factor, then we
    do not factor. If we do factor the accumulator, then this function returns a
    tuple of the two largest axes to reduce over.

    Args:
        shape: a Shape

    Returns:
        None or a tuple of ints
    """
    if len(shape) < 2:
        return None
    sorted_dims = torch.argsort(shape)
    return int(sorted_dims[-2]), int(sorted_dims[-1])


def _clip_log_abs(v, scale=1.0):
    mag = torch.log(1e-8 + torch.abs(v * scale))
    return torch.clip(mag, -5, 5) * 0.5


def _sorted_values(dd):
    return list(zip(*sorted(dd.items(), key=lambda x: x[0])))[1]


class BufferLossAccumulators:
    """Rolling accumulator for loss values."""

    def __init__(self):
        pass

    def init(self, num_steps):
        halflife = torch.logspace(1, torch.log10(num_steps), 10)
        decays = torch.exp(-1. / halflife)
        return {
            "means": torch.zeros((len(decays),), dtype=torch.float32),
            "iteration": torch.tensor(0, dtype=torch.int32),
            "running_min": 999999999999. * torch.ones((len(decays),), dtype=torch.float32),
            "decays": decays,
        }

    def update(self, state, loss):
        """Update the state with a new loss."""
        jdecays = state["decays"]
        cor_mean = state["means"] / (1 - jdecays ** (state["iteration"] + 1))
        approx_max = torch.max(cor_mean)
        approx_max = torch.where(state["iteration"] == 0, loss, approx_max)
        loss = torch.minimum(torch.abs(approx_max) * 2, loss)

        means = state["means"] * jdecays + loss * (1. - jdecays)

        cor_mean = means / (1 - jdecays ** (state["iteration"] + 1))
        running_min = torch.minimum(state["running_min"], cor_mean)

        return {
            "means": means,
            "iteration": state["iteration"] + 1,
            "running_min": running_min,
            "decays": state["decays"],
        }

    def features(self, state):
        """Compute features to pass to NN from state."""
        jdecays = state["decays"]
        cor_mean = state["means"] / (1 - jdecays ** (state["iteration"]))
        # longest running decay
        approx_max = cor_mean[1:]
        cor_mean = cor_mean[0:-1]
        running_min = state["running_min"][0:-1]

        den = torch.maximum(1e-8, (approx_max - running_min))
        pre_center = (cor_mean - running_min) / den
        feature1 = (pre_center - 1.0)
        feature1 = torch.clamp(feature1, -1, 1)
        # first couple features are bad.
        return torch.where(state["iteration"] <= 2, feature1 * 0, feature1)


class State:
    """Inner state of learned optimizer."""

    def __init__(self, params, rms_rolling, mom_rolling, fac_rolling, iteration, state, num_steps, loss_buffer):
        self.params = params
        self.rms_rolling = rms_rolling
        self.mom_rolling = mom_rolling
        self.fac_rolling = fac_rolling
        self.iteration = iteration
        self.state = state
        self.num_steps = num_steps
        self.loss_buffer = loss_buffer


def _safe_rsqrt(x):
    return torch.rsqrt(torch.maximum(x, 1e-9))


def _second_moment_normalizer(x, axis, eps=1e-5):
    return x * torch.rsqrt(eps +
                           torch.mean(torch.square(x), dim=axis, keepdim=True))
