import argparse
import copy
import functorch
import torch
import importlib
from networks import get_pretrained_net_fixed
from learned_optimizer import OptimizerTrainer, fine_tune_func_single, fine_tune

def test_fine_tune_func_single(args):
    print("Testing fine_tune_func_single...")
    net = copy.deepcopy(get_pretrained_net_fixed(ckpt_path=args.ckpt_path, train=True))
    train_x = torch.randn(args.meta_batch_size // 2 * args.batch_size, 28, 28, dtype=torch.float32)
    train_y = torch.randint(10, (args.meta_batch_size // 2 * args.batch_size,))
    test_x = torch.randn(args.meta_batch_size // 2 * args.batch_size, 28, 28, dtype=torch.float32)
    test_y = torch.randint(10, (args.meta_batch_size // 2 * args.batch_size,))

    optimizer_func_module = importlib.import_module(f"optimizers_func")
    optimizer_obj_func = getattr(optimizer_func_module, args.optimizer_name)
    func_net, net_params = functorch.make_functional(net)
    meta_params = optimizer_obj_func.get_init_meta_params()
    test_losses_func = fine_tune_func_single(optimizer_obj_func, args.inner_steps, args.inner_lr, func_net, net_params, meta_params, train_x, train_y, test_x, test_y)

    optimizer_module = importlib.import_module(f"optimizers")
    optimizer_obj = getattr(optimizer_module, args.optimizer_name)
    meta_params = optimizer_obj.get_init_meta_params(num_features=None)
    # optimizer_obj, inner_steps, inner_lr, features, _net, meta_params, train_images, train_labels, test_images, test_labels
    test_losses = fine_tune(optimizer_obj, args.inner_steps, args.inner_lr, None, net, meta_params, train_x, train_y, test_x, test_y)

    print("Test losses func", torch.tensor(test_losses_func, dtype=torch.float64))
    print("Test losses", test_losses)
    assert torch.allclose(torch.tensor(test_losses_func, dtype=torch.float64), torch.tensor(test_losses, dtype=torch.float64))
    print("Passed fine-tune function test.")

def test_outer_step_parallel(args):
    print("Testing outer_step_parallel...")
    net = copy.deepcopy(get_pretrained_net_fixed(ckpt_path=args.ckpt_path, train=True))
    train_x, train_y = torch.randn(args.meta_batch_size // 2, args.batch_size, 1, 28, 28, dtype=torch.float32), torch.randint(10, (args.meta_batch_size // 2, args.batch_size))
    test_x, test_y = torch.randn(args.meta_batch_size // 2, args.batch_size, 1, 28, 28,
                                   dtype=torch.float32), torch.randint(10, (args.meta_batch_size // 2, args.batch_size))

    args.run_parallel = False
    opt_trainer_iter = OptimizerTrainer(args)
    epsilon = torch.tensor([0.05, 0.05, 0.05, 0.05])
    meta_params_iter, grad_mean_iter = opt_trainer_iter.outer_loop_step_iter(net, epsilon, train_x, train_y, test_x, test_y)

    args.run_parallel = True
    opt_trainer_parallel = OptimizerTrainer(args)
    epsilon_batched = torch.tensor([[0.05, 0.05, 0.05, 0.05] for _ in range(args.meta_batch_size // 2)]).cuda()
    meta_params_parallel, grad_mean_parallel = opt_trainer_parallel.outer_loop_step_parallel(net, epsilon_batched, train_x, train_y, test_x, test_y)
    grad_mean_parallel = grad_mean_parallel.detach().cpu()

    print("Outer loop step iter", grad_mean_iter)
    print("Outer loop step parallel", grad_mean_parallel)

    print('meta params iter', meta_params_iter)
    print('meta params parallel', meta_params_parallel)
    assert torch.allclose(grad_mean_iter, grad_mean_parallel)
    print("Passed outer step function test.")

if __name__ == '__main__':
    args = argparse.Namespace(
        batch_size=128,
        ckpt_path='/iris/u/cchoi1/robust-optimizer/mnist/ckpts',
        data_dir='/iris/u/cchoi1/Data',
        features=None,
        ft_id_dist='brightness',
        ft_id_ood=False,
        ft_ood_dist='impulse_noise',
        inner_lr=0.1,
        inner_steps=10,
        l2_lambda=None,
        meta_batch_size=20,
        meta_loss_avg_w=0.0,
        meta_loss_final_w=1.0,
        meta_lr=0.003,
        meta_steps=160,
        method='ours',
        noise_std=1.0,
        num_epochs=40,
        num_nets=1,
        num_seeds=3,
        num_workers=0,
        optimizer_name='LayerSGD',
        patience=3,
        run_parallel=False,
        test_dist='impulse_noise',
        val='ood',
        val_freq=10,
        val_meta_batch_size=100
    )
    test_fine_tune_func_single(args)
    test_outer_step_parallel(args)