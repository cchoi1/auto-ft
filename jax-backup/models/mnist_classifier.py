
# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MNIST example.
Library file which executes the training and evaluation loop for MNIST.
The data is loaded using tensorflow_datasets.
"""

# See issue #620.
# pytype: disable=wrong-keyword-args

import configparser
import datasets
from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow_datasets as tfds
from flax.training import checkpoints, train_state


class MLP(nn.Module):
    """A simple MLP model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(784)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x


@jax.jit
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        # one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def get_datasets():
    """Load MNIST train and test data into memory."""
    # ds_builder = tfds.builder('mnist')
    # ds_builder.download_and_prepare()
    # train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    # test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    # train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    # breakpoint()
    # train_ds['image'] = train_ds['image'].reshape(train_ds['image'].shape[0], -1)
    # breakpoint()
    # test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    # test_ds['image'] = test_ds['image'].reshape(test_ds['image'].shape[0], -1)
    train_images, train_labels, test_images, test_labels = datasets.mnist()
    train_ds = {}
    test_ds = {}
    train_ds['image'] = train_images
    train_ds['label'] = train_labels
    test_ds['image'] = test_images
    test_ds['label'] = test_labels
    return train_ds, test_ds


def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    mlp = MLP()
    # params = mlp.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    params = mlp.init(rng, jnp.ones([1, 28 * 28]))['params']
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(
        apply_fn=mlp.apply, params=params, tx=tx)


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> train_state.TrainState:
    """Execute model training and evaluation loop.
    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.
    Returns:
      The train state (which includes the `.params`).
    """
    train_ds, test_ds = get_datasets()
    rng = jax.random.PRNGKey(0)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(state, train_ds,
                                                        config.batch_size,
                                                        input_rng)
        _, test_loss, test_accuracy = apply_model(state, test_ds['image'],
                                                  test_ds['label'])

        logging.info(
            'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
            % (epoch, train_loss, train_accuracy * 100, test_loss,
               test_accuracy * 100))
        print(
            'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
            % (epoch, train_loss, train_accuracy * 100, test_loss,
               test_accuracy * 100))

        summary_writer.scalar('train_loss', train_loss, epoch)
        summary_writer.scalar('train_accuracy', train_accuracy, epoch)
        summary_writer.scalar('test_loss', test_loss, epoch)
        summary_writer.scalar('test_accuracy', test_accuracy, epoch)
        # print(f"Before checkpoint test accuracy: {test_accuracy}")

        CKPT_DIR = "ckpts/mnist"
        ckpt_path = checkpoints.save_checkpoint(
          ckpt_dir=CKPT_DIR,
          target=state,
          step=epoch,
          overwrite=True,
          keep=1)
        state = checkpoints.restore_checkpoint(
          ckpt_dir=CKPT_DIR,
          target=state)
        # _, test_loss, test_accuracy = apply_model(state, test_ds['image'],
        #                                           test_ds['label'])
        # print(f"After checkpoint test accuracy: {test_accuracy}")


    summary_writer.flush()
    return state

def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.learning_rate = 0.001
  config.momentum = 0.9
  config.batch_size = 128
  config.num_epochs = 20
  return config


if __name__ == '__main__':
    config = get_config()
    train_and_evaluate(config, workdir='./tensorboard')