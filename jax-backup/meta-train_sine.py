import jax
import argparse
import pickle
import os
import numpy as np

import jax
from flax.training import checkpoints
from flax.core import freeze, unfreeze
from learned_optimization.optimizers import base as opt_base
from jax.example_libraries import optimizers

from learned_optimizer import PerParamMLP
from tasks import image_mlp, sine
from tasks.sine import SineTask



class OptimizerTrainer:
    def __init__(self, lopt, pretrain_task, finetune_task, checkpoint_path):
        self.lopt = lopt
        self.pretrain_task = pretrain_task
        self.finetune_task = finetune_task
        self.checkpoint_path = checkpoint_path

    def save_model(self, params):
        model_file = f"{self.checkpoint_path}_pretrained.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(params, f)

    def pretrain(self, num_epochs):
        key = jax.random.PRNGKey(0)
        key1, key = jax.random.split(key)
        params = self.pretrain_task.init(key1)
        opt = opt_base.SGD(1e-2)
        opt_state = opt.init(params)
        loss_value_and_grad = jax.jit(jax.value_and_grad(self.pretrain_task.loss))
        for i in range(num_epochs):
            batch = self.pretrain_task.datasets
            params = opt.get_params(opt_state)
            loss, grad = loss_value_and_grad(params, key1, batch["x"], batch["y"])
            opt_state = opt.update(opt_state, grad, loss)
        print(f"Final pretraining loss: {loss}")
        return params, loss

    def meta_loss(self, theta, key, batch, pretrain_param):
        """
        Computes loss of applying a learned optimizer to a given task for some number of steps.
        """
        opt = self.lopt.opt_fn(theta)
        key1, key = jax.random.split(key)
        param = self.finetune_task.init(key1)
        # TODO: initiate self.finetune_task._mod with pretrained params in Haiku, and replace the following line
        param, loss = self.pretrain(num_epochs=50)
        opt_state = opt.init(param)

        for i in range(4):
            param = opt.get_params(opt_state)
            key1, key = jax.random.split(key)
            l, grad = jax.value_and_grad(self.finetune_task.loss)(param, key1, batch['x'], batch['y'])
            opt_state = opt.update(opt_state=opt_state, grads=grad, model_state=l)

        param, state = opt.get_params_state(opt_state)
        key1, key = jax.random.split(key)
        final_loss = self.finetune_task.loss(param, key1, batch['x'], batch['y'])
        return final_loss

    # def meta_loss(self, finetune_tasks, theta, key, batch, pretrain_param):
    #     task_losses = jax.vmap(self.task_loss, in_axes=(None, 0, 0))(finetune_tasks, theta, key, x_tasks, y_tasks)
    #     return jax.numpy.mean(task_losses)

    def meta_train(self, pretrain_param):
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
            batch = self.finetune_task.datasets
            key, key1 = jax.random.split(key)
            theta = theta_opt.get_params(theta_opt_state)
            ml, meta_grad = meta_value_and_grad(theta, key, batch, pretrain_param)
            theta_opt_state = theta_opt.update(theta_opt_state, meta_grad, ml)
            if i % 100 == 0:
                print(f"Meta-Train Loss: {ml:.2f}")

def run_expt(args):
    lopt = PerParamMLP()
    pretrain_task = SineTask(N=1000, amplitude=1, phase=1, vertical_shift=0, x_min=-1, x_max=1)
    finetune_task = SineTask(N=1000, amplitude=0.5, phase=1, vertical_shift=0, x_min=-1, x_max=0)
    # # C is randomly sampled
    # finetune_tasks = []
    # for i in range(args.num_tasks):
    #     # Randomly sample C
    #     C = np.random.uniform(0, 1)
    #     # C * sin(x)
    #     finetune_tasks.append(SineTask(N=1000, amplitude=C, phase=1, vertical_shift=0, x_min=-1, x_max=0))

    key = jax.random.PRNGKey(0)
    theta = lopt.init(key)
    jax.tree_util.tree_map(lambda x: (x.shape, x.dtype), theta)
    ckpt_path = '/iris/u/cchoi1/robust-optimizer/models/ckpts/sine'
    trainer = OptimizerTrainer(lopt, pretrain_task, finetune_task, ckpt_path)
    # uncomment below line to get actual printed final pretrain loss (otherwise just shows jax tracer)
    pretrain_param = trainer.pretrain(num_epochs=50)
    trainer.meta_train(pretrain_param)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default='/iris/u/cchoi1/robust-optimizer/models/ckpts/mnist',
                        help="directory to pretrained model checkpoint")
    parser.add_argument("--num_tasks", type=int, default=10)
    args = parser.parse_args()
    run_expt(args)