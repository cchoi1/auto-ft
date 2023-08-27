import os
import time

import src.datasets as datasets
import torch
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ImageClassifier
from src.models.utils import cosine_lr, LabelSmoothing
from src.losses.layerloss import LayerLoss


def finetune(args, model, loss_fn, optimizer, dataloader, input_key, print_every):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."

    model.train()
    params = list(model.parameters())
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.inner_steps)

    iter = 0
    while iter < args.inner_steps:
        for batch in dataloader:
            start_time = time.time()
            scheduler(iter)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time
            # print(f"Data (t) {data_time:.3f}", flush=True)

            if isinstance(loss_fn, LayerLoss):
                loss, _ = loss_fn(inputs, labels, model)
            else:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            batch_time = time.time() - start_time
            # print(f"Batch (t) {batch_time:.3f}", flush=True)

            if print_every is not None and iter % print_every == 0:
                percent_complete = 100 * iter / args.inner_steps
                print(f"Train Iter: {iter}/{args.inner_steps} [{percent_complete:.0f}% ]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True)

            del inputs, labels, loss
            torch.cuda.empty_cache()
            iter += 1
    del dataloader

    # Evaluate
    if args.method != "autoft":
        args.current_epoch = iter
        eval_results = evaluate(model.module, args)

    if args.save is not None and args.method != "autoft":
        os.makedirs(args.save, exist_ok=True)
        model_path = os.path.join(args.save, f'checkpoint_{args.inner_steps}.pt')
        print('Saving model to', model_path)
        model.module.save(model_path)
        optim_path = os.path.join(args.save, f'optim_{args.inner_steps}.pt')
        torch.save(optimizer.state_dict(), optim_path)
        if args.save is not None:
            return model_path

    return model.module