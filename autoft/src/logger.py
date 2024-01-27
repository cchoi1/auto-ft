import os
import logging


def setup_logging(args, logger):
    """
    Sets up logging for the training process.

    :param args: Namespace containing script arguments.
    :param logger: The logger instance to configure.
    :return: Configured logger instance.
    """

    # Create save directory
    save_dir = os.path.join(args.save, args.id, args.method)
    os.makedirs(save_dir, exist_ok=True)

    # Construct method name based on arguments
    method_name_parts = [f"ood{args.ood}"]
    if args.optuna_sampler != "TPESampler":
        method_name_parts.append(args.optuna_sampler)
    if args.layerwise_loss:
        method_name_parts.append("layerwiseloss")
    if args.layerwise_opt:
        method_name_parts.append("layerwiseopt")
    method_name_parts.append(''.join([l[0] for l in args.losses]))
    method_name = '_'.join(method_name_parts)

    # Construct run details string
    val_set = f"no{args.num_ood_hp_examples}"
    run_details = f"{val_set}_hopt{args.hopt_evals}_is{args.inner_steps}_ep{args.ft_epochs}_bs{args.batch_size}_wd{args.wd}_lr{args.lr}_runs{args.runs}_{args.model}"

    # Update args.save with the new path
    args.save = os.path.join(save_dir, method_name, run_details)
    os.makedirs(args.save, exist_ok=True)

    # Setup logging path
    logging_path = os.path.join("logs", args.save)
    print(f"\nModel Save Path: {args.save}")
    print(f"\nLogging Path: {logging_path}\n")
    os.makedirs(logging_path, exist_ok=True)

    # Configure logging
    log_filename = os.path.join(logging_path, "log.log")
    logging.basicConfig(filename=log_filename, format='%(asctime)s %(message)s', filemode='w')
    logger.setLevel(logging.INFO)

    return logger

# import logging
# import os
#
# def setup_logging(args, logger):
#     save_dir = os.path.join(args.save, args.id, args.method)
#     os.makedirs(save_dir, exist_ok=True)
#
#     method_name = f"ood{args.ood}"
#     if args.optuna_sampler != "TPESampler":
#         method_name += f"_{args.optuna_sampler}"
#     loss_type = "_layerwiseloss" if args.layerwise_loss else ""
#     opt_type = "_layerwiseopt" if args.layerwise_opt else ""
#     method_name += loss_type + opt_type
#     method_name += "_" + "".join([l[0] for l in args.losses])
#     val_set = f"no{args.num_ood_hp_examples}"
#     run_details = (f"{val_set}_hopt{args.hopt_evals}_is{args.inner_steps}_ep{args.ft_epochs}_bs{args.batch_size}_"
#                    f"wd{args.wd}_lr{args.lr}_runs{args.runs}_{args.model}")
#     args.save = os.path.join(save_dir, method_name, run_details)
#
#     os.makedirs(args.save, exist_ok=True)
#     logging_path = os.path.join("logs", args.save)
#     print(f"\nModel Save Path: {args.save}")
#     print(f"\nLogging Path: {logging_path}\n")
#     os.makedirs(logging_path, exist_ok=True)
#     log_filename = logging_path + "/log.log"
#     logging.basicConfig(filename=log_filename, format='%(asctime)s %(message)s', filemode='w')
#     logger.setLevel(logging.INFO)
#
#     return logger