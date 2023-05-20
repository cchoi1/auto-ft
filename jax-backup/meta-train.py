import flax
from typing import Any
import haiku as hk
import jax.numpy as jnp
import jax

from learned_optimization.learned_optimizers import base as lopt_base
from learned_optimization.optimizers import base as opt_base
from learned_optimization.tasks.fixed import image_mlp

from learned_optimizer import PerParamMLP

class OptimizerTrainer:
    def __init__(self, lopt, task):
        self.lopt = lopt
        self.task = task

    def meta_loss(self, theta, key, batch):
        """
        Computes loss of applying a learned optimizer to a given task for some number of steps.
        """
        opt = self.lopt.opt_fn(theta)
        key1, key = jax.random.split(key)
        param = self.task.init(key1)
        opt_state = opt.init(param)
        for i in range(4):
            param = opt.get_params(opt_state)
            key1, key = jax.random.split(key)
            l, grad = jax.value_and_grad(self.task.loss)(param, key1, batch)
            opt_state = opt.update(opt_state=opt_state, grads=grad, model_state=l)

        param, state = opt.get_params_state(opt_state)
        key1, key = jax.random.split(key)
        final_loss = self.task.loss(param, key1, batch)
        return final_loss

    def meta_train(self):
        """
        Optimizes theta, the weights of the learned optimizer.
        """
        # Get the meta-gradient -- gradients wrt weights of the learned optimizer
        meta_value_and_grad = jax.jit(jax.value_and_grad(self.meta_loss))
        theta_opt = opt_base.Adam(1e-2)

        key = jax.random.PRNGKey(0)
        theta = self.lopt.init(key)
        theta_opt_state = theta_opt.init(theta)

        for i in range(2000):
            batch = next(self.task.datasets.train)
            key, key1 = jax.random.split(key)
            theta = theta_opt.get_params(theta_opt_state)
            ml, meta_grad = meta_value_and_grad(theta, key, batch)
            theta_opt_state = theta_opt.update(theta_opt_state, meta_grad, ml)
            if i % 100 == 0:
                print(f"Meta-Train Loss: {ml:.2f}")

def run_expt():
    lopt = PerParamMLP()
    # TODO: Implement an MNIST-C task and replace the below task
    task = image_mlp.ImageMLP_Mnist_128x128x128_Relu()
    key = jax.random.PRNGKey(0)
    theta = lopt.init(key)
    jax.tree_util.tree_map(lambda x: (x.shape, x.dtype), theta)
    trainer = OptimizerTrainer(lopt, task)
    trainer.meta_train()


if __name__ == '__main__':
    run_expt()