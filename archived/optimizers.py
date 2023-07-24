class PerParamSGD(Optimizer):
    """meta-params: pre-sigmoid lr_multiplier per parameter."""

    def __init__(self, meta_params, net, inp_info=None, lr=required):
        # assert meta_params.numel() == 4
        defaults = dict(lr=lr)
        params = net.parameters()
        super().__init__(params, defaults)
        self.lr_multiplier = torch.sigmoid(meta_params).to(device)

    @staticmethod
    def get_init_meta_params(inp_info):
        """tensor_shapes: a list of shapes of each tensor in the network."""
        num_params = sum([t.numel() for t in inp_info["tensor_shapes"]])
        return torch.zeros(num_params)

    @staticmethod
    def get_noise(inp_info):
        """tensor_shapes: a list of shapes of each tensor in the network."""
        num_params = sum([t.numel() for t in inp_info["tensor_shapes"]])
        return torch.randn(num_params)

    def step(self, curr_loss=None, iter=None, iter_frac=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        start = 0
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                d_p = p.grad.data

                end = start + p.numel()
                p_lr_multiplier = self.lr_multiplier[start:end].view(p.shape)
                start = end

                p_lr = group["lr"] * p_lr_multiplier
                p.data.add_(d_p * -p_lr)
        assert end == self.lr_multiplier.numel()
        return loss