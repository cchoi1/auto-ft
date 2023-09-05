import math
import torch
import torch_xla.core.xla_model as xm

def get_sampler(dataset, pointwise_weights, shuffle):
    sampler_kwargs = {"dataset": dataset, "shuffle": shuffle}
    if xm.xrt_world_size() > 1:
        sampler_kwargs["num_replicas"] = xm.xrt_world_size()
        sampler_kwargs["rank"] = xm.get_ordinal()
    elif torch.cuda.world_size() > 1:
        sampler_kwargs["num_replicas"] = torch.cuda.world_size()
        sampler_kwargs["rank"] = torch.cuda.current_device()

    if pointwise_weights is not None:
        sampler_kwargs["weights"] = pointwise_weights
        sampler = DistributedWeightedRandomSampler(**sampler_kwargs)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(**sampler_kwargs)

    return sampler

class DistributedWeightedRandomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, weights, num_replicas=None, rank=None):
        # Assuming weights is a tensor of same size as dataset
        self.weights = weights
        self.dataset = dataset

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            self.num_replicas = torch.distributed.get_world_size()
        else:
            self.num_replicas = num_replicas

        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = rank

        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        # This will only be a subset of the entire range, specific to this rank/process
        self.indices = list(range(len(self.dataset)))[self.rank:self.total_size:self.num_replicas]
        self.weighted_sampler = torch.utils.data.WeightedRandomSampler(weights[self.indices], len(self.indices))

    def __iter__(self):
        return iter([self.indices[i] for i in self.weighted_sampler])

    def __len__(self):
        return len(self.indices)
