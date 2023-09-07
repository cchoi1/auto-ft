import torch
import torch_xla.core.xla_model as xm

def get_sampler(dataset, shuffle):
    sampler_kwargs = {"dataset": dataset, "shuffle": shuffle}
    if xm.xrt_world_size() > 1:
        sampler_kwargs["num_replicas"] = xm.xrt_world_size()
        sampler_kwargs["rank"] = xm.get_ordinal()
    elif torch.cuda.device_count() > 1:
        sampler_kwargs["num_replicas"] = torch.cuda.device_count()
        sampler_kwargs["rank"] = torch.cuda.current_device()

    if xm.xrt_world_size() > 1 or torch.cuda.device_count() > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(**sampler_kwargs)
    else:
        sampler = None

    return sampler