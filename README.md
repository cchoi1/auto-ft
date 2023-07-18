# Meta-Learning a Robust Optimizer for Fine-Tuning

Relevant repositories

ES
- https://github.com/openai/evolution-strategies-starter
- https://github.com/SimonBlanke/Gradient-Free-Optimizers
- https://github.com/google/learned_optimization/blob/main/learned_optimization/outer_trainers/full_es.py

Torch optimizers
- https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
- https://github.com/rahulkidambi/AccSGD/blob/master/AccSGD.py
- https://github.com/kach/gradient-descent-the-ultimate-optimizer/blob/main/src/gradient_descent_the_ultimate_optimizer/gdtuo.py
- https://github.com/clovaai/AdamP/blob/master/adamp/adamp.py

VeLO Features:
- Code: https://github.com/google/learned_optimization/blob/687e72e7b5596dfb80c5196fd51f43058899edd9/learned_optimization/research/general_lopt/hyper_v2.py#L80
- Appendix B of paper

Documentation

functorch documentation
- https://pytorch.org/functorch/stable/

functorch examples
- MAML: https://github.com/metaopt/torchopt/blob/main/examples/FuncTorch/maml_omniglot_vmap.py
- Model Ensemble: https://github.com/metaopt/torchopt/blob/main/examples/FuncTorch/parallel_train_torchopt.py 

torch.multiprocessing
- https://pytorch.org/docs/stable/notes/multiprocessing.html

PyTorch DDP
- https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

PyTorch on XLA devices/TPUs
- https://pytorch.org/xla/release/2.0/index.html#how-to-do-distributeddataparallel
