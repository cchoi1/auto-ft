import logging
import os

def setup_logging(args, logger):
    save_dir = os.path.join(args.save, args.id, args.method)
    os.makedirs(save_dir, exist_ok=True)

    if args.method == "autoft":
        loss_type = "layerwiseloss" if args.layerwise_loss else ""
        opt_type = "layerwiseopt" if args.layerwise_opt else ""
        method_name = f"ood{args.ood}"
        if args.layerwise_loss:
            method_name += "_lloss"
        if args.layerwise_opt:
            method_name += "_lopt"
        if args.unlabeled_id is not None:
            method_name += "_unlabeled"
        if args.ft_data is not None:
            method_name += "_flyp"
        if args.optuna_sampler != "TPESampler":
            method_name += f"_{args.optuna_sampler}"
        method_name += loss_type + opt_type
        method_name += "_" + "".join([l[0] for l in args.losses])
        val_set = f"no{args.num_ood_hp_examples}"
        if args.val_mini_batch_size is not None:
            val_set += f"_mbs{args.val_mini_batch_size}"
        run_details = f"{val_set}_nou{args.num_ood_unlabeled_examples}_afep{args.autoft_epochs}_is{args.inner_steps}_ftep{args.ft_epochs}_bs{args.batch_size}_wd{args.wd}_lr{args.lr}_run{args.run}_seed{args.seed}"
        args.save = os.path.join(save_dir, method_name, run_details)
    elif args.method == "ft-id-ood":
        method_name = f"ood{args.ood}"
        if args.num_ood_unlabeled_examples is not None:
            method_name += "_unlabeled"
        run_details = f"no{args.num_ood_hp_examples}_nou{args.num_ood_unlabeled_examples}_ftep{args.ft_epochs}_bs{args.batch_size}_wd{args.wd}_lr{args.lr}_run{args.run}_seed{args.seed}"
        args.save = os.path.join(save_dir, method_name, run_details)
    elif args.method == "ft-id":
        run_details = f"ftep{args.ft_epochs}_bs{args.batch_size}_wd{args.wd}_lr{args.lr}_run{args.run}_seed{args.seed}"
        args.save = os.path.join(save_dir, run_details)

    os.makedirs(args.save, exist_ok=True)
    logging_path = os.path.join("logs", args.save)
    print(f"\nMODEL SAVE PATH: {args.save}")
    print(f"\nLOGGING PATH: {logging_path}\n")
    os.makedirs(logging_path, exist_ok=True)
    log_filename = logging_path + "/log.log"
    logging.basicConfig(filename=log_filename,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger.setLevel(logging.INFO)
    return logger