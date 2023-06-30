import copy
import importlib
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from torch import nn

from mnist import load_dataset
from networks import get_pretrained_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def fine_tune(_net, meta_params, train_images, train_labels, test_images, test_labels, optimizer_obj, inner_steps=10,
              inner_lr=1e-1):
    """Fine-tune net on ft_data, and return net, test accuracies, and test losses."""
    net = copy.deepcopy(_net)
    inner_opt = optimizer_obj(meta_params, net, lr=inner_lr)
    loss_fn = nn.CrossEntropyLoss()
    test_accs = []
    test_losses = []
    for _ in range(inner_steps):
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        preds = net(train_images)
        loss = loss_fn(preds, train_labels)
        inner_opt.zero_grad()
        loss.backward()
        inner_opt.step()

        test_images, test_labels = test_images.to(device), test_labels.to(device)
        output = net(test_images)
        test_loss = loss_fn(output, test_labels)
        test_losses.append(test_loss.item())
        preds = torch.argmax(output.data, -1)
        total = test_labels.size(0)
        correct = (preds == test_labels).sum().item()
        test_acc = correct / total
        test_accs.append(test_acc)

    return net, meta_params, np.array(test_accs), np.array(test_losses)


class OptimizerTrainer:
    def __init__(self, args):
        self.ft_distribution = args.ft_distribution
        self.data_dir = args.data_dir
        self.ckpt_path = args.ckpt_path

        optimizer_module = importlib.import_module(f"optimizers")
        self.optimizer_obj = getattr(optimizer_module, args.optimizer_name)
        self.meta_params = self.optimizer_obj.get_init_meta_params()
        self.meta_optimizer = torch.optim.SGD([self.meta_params], lr=args.meta_lr)
        self.train_loader, self.id_val_loader = load_dataset(root_dir=self.data_dir, dataset=args.ft_distribution)
        self.test_loader, self.ood_val_loader = load_dataset(root_dir=self.data_dir, dataset=args.test_distribution)

        # Inner Loop Hyperparameters
        self.val_meta_batch_size = args.val_meta_batch_size
        self.inner_steps = args.inner_steps
        self.inner_lr = args.inner_lr
        self.train_N = args.train_N

        # Outer Loop Hyperparameters
        self.meta_lr = args.meta_lr
        self.meta_batch_size = args.meta_batch_size
        self.noise_std = args.noise_std
        self.meta_loss_avg_w = args.meta_loss_avg_w
        self.meta_loss_final_w = args.meta_loss_final_w

        self.finetune = partial(
            fine_tune,
            optimizer_obj=self.optimizer_obj,
            inner_steps=self.inner_steps,
            inner_lr=self.inner_lr,
        )

    def validation(self, repeat):
        accs = defaultdict(list)
        losses = defaultdict(list)
        for _ in range(repeat):
            net = get_pretrained_net(ckpt_path=self.ckpt_path, train=False)
            train_images, train_labels = next(iter(self.train_loader))
            val_images, val_labels = next(iter(self.ood_val_loader))
            _, _, val_accs, val_losses = self.finetune(net, self.meta_params, train_images, train_labels, val_images,
                                                       val_labels)
            accs["ood"].append(val_accs[-1])
            losses["ood"].append(val_losses[-1])
        for _ in range(repeat):
            net = get_pretrained_net(ckpt_path=self.ckpt_path, train=True)
            train_images, train_labels = next(iter(self.train_loader))
            val_images, val_labels = next(iter(self.id_val_loader))
            _, _, train_accs, train_losses = self.finetune(net, self.meta_params, train_images, train_labels,
                                                           val_images, val_labels)
            accs["id"].append(train_accs[-1])
            losses["id"].append(train_losses[-1])
        id_val_str = f"ID Val acc: {np.mean(accs['id']):.4f} +- {np.std(accs['id']):.4f}"
        ood_val_str = (
            f"OOD Val acc: {np.mean(accs['ood']):.4f} +- {np.std(accs['ood']):.4f}"
        )
        print(id_val_str, '|', ood_val_str)
        return accs

    def outer_loop_step(self):
        """Perform one outer loop step. meta_batch_size tasks with antithetic sampling."""
        grads = []
        for _ in range(self.meta_batch_size // 2):
            net = get_pretrained_net(ckpt_path=self.ckpt_path, train=True)
            epsilon = (
                    self.optimizer_obj.get_noise() * self.noise_std
            )  # Antithetic sampling
            mp_plus_epsilon = self.meta_params + epsilon
            mp_minus_epsilon = self.meta_params - epsilon
            train_images, train_labels = next(iter(self.train_loader))
            test_images, test_labels = next(iter(self.ood_val_loader))
            _, _, _, losses_plus = self.finetune(net, mp_plus_epsilon, train_images, train_labels, test_images,
                                                 test_labels)
            _, _, _, losses_minus = self.finetune(net, mp_minus_epsilon, train_images, train_labels, test_images,
                                                  test_labels)
            loss_diff = losses_plus - losses_minus
            objective = (
                    loss_diff[-1] * self.meta_loss_final_w
                    + loss_diff.mean() * self.meta_loss_avg_w
            )
            grads.append(objective * epsilon / self.noise_std / 2)
        grads_mean = torch.stack(grads).mean(dim=0)

        self.meta_optimizer.zero_grad()
        self.meta_params.grad = grads_mean
        self.meta_optimizer.step()

        return self.meta_params
