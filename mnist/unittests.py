import argparse
import copy
import functorch
import torch
import importlib
from networks import pretrain_nets, get_pretrained_net_fixed
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
    meta_params = optimizer_obj.get_init_meta_params()
    test_losses = fine_tune(optimizer_obj, args.inner_steps, args.inner_lr, net, meta_params, train_x, train_y, test_x, test_y)

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