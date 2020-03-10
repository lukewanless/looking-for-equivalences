from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch
import re
import logging
import os
import pandas as pd
import numpy as np
import random


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def evaluate(eval_dataset, hyperparams, model):

    verbose = hyperparams["verbose"]
    disable = False if verbose else True

    per_gpu_eval_batch_size = hyperparams["per_gpu_eval_batch_size"]
    n_gpu = hyperparams["n_gpu"]
    device = hyperparams["device"]
    model_type = hyperparams["model_type"]

    eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=eval_batch_size)
    # multi-gpu eval
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", eval_batch_size)
    preds = None
    out_label_ids = None

    results = {"label": [],
               "prediction": []}
    all_loss = []

    for batch in tqdm(eval_dataloader,
                      desc="Evaluating",
                      disable=disable):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
            # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            if model_type != "distilbert":
                inputs["token_type_ids"] = (batch[2] if model_type in [
                    "bert", "xlnet", "albert"] else None
                )
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            all_loss.append(loss.item())

            if preds is None:
                preds = logits.detach().cpu().numpy()
                preds = np.argmax(preds, axis=1)

                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                new_preds = logits.detach().cpu().numpy()
                new_preds = np.argmax(new_preds, axis=1)
                preds = np.append(preds, new_preds, axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    results["label"] = out_label_ids
    results["prediction"] = preds
    return np.mean(all_loss), pd.DataFrame(results)


def train(train_dataset,
          train_dataset_to_eval,
          dev_dataset_to_eval,
          model, tokenizer, hyperparams):

    verbose = hyperparams["verbose"]
    disable = False if verbose else True

    local_rank = hyperparams["local_rank"]
    per_gpu_train_batch_size = hyperparams["per_gpu_train_batch_size"]
    n_gpu = hyperparams["n_gpu"]
    max_steps = hyperparams["max_steps"]
    num_train_epochs = hyperparams["num_train_epochs"]
    gradient_accumulation_steps = hyperparams["gradient_accumulation_steps"]
    weight_decay = hyperparams["weight_decay"]
    learning_rate = hyperparams["learning_rate"]
    adam_epsilon = hyperparams["adam_epsilon"]
    warmup_steps = hyperparams["warmup_steps"]
    seed = hyperparams["random_state"]
    device = hyperparams["device"]
    model_type = hyperparams["model_type"]
    max_grad_norm = hyperparams["max_grad_norm"]

    save_steps = hyperparams['save_steps']

    base_output_dir = hyperparams["output_dir"]
    train_log_path = os.path.join(base_output_dir, "train_log.csv")
    eval_log_path = os.path.join(base_output_dir, "eval_log.csv")

    fp16_opt_level = hyperparams["fp16_opt_level"]
    fp16 = hyperparams["fp16"]

    model_name_or_path = hyperparams["model_name_or_path"]
    opt_path = os.path.join(model_name_or_path, "optimizer.pt")
    sche_path = os.path.join(model_name_or_path, "scheduler.pt")

    training_logs = {"loss": [],
                     "learning_rate": []}

    eval_logs = {"step": [],
                 "train_acc": [],
                 "dev_acc": [],
                 "train_loss": [],
                 "dev_loss": []}

    train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)

    if local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=train_batch_size)

    if max_steps > 0:
        t_total = max_steps
        num_train_epochs = max_steps // (len(train_dataloader) //
                                         gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=learning_rate,
                      eps=adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(opt_path) and os.path.isfile(sche_path):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(opt_path))
        scheduler.load_state_dict(torch.load(sche_path))

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", num_train_epochs)
    logging.info(
        "  Instantaneous batch size per GPU = %d",
        per_gpu_train_batch_size)
    logging.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", train_batch_size
                 * gradient_accumulation_steps
                 * (torch.distributed.get_world_size() if local_rank != -1 else 1))
    logging.info(
        "  Gradient Accumulation steps = %d",
        gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(model_name_or_path) and model_name_or_path.find(
            "checkpoints") > 0:
        # set global_step to gobal_step of last saved checkpoint from model
        # path
        global_step = int(model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) //
                                         gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // gradient_accumulation_steps)

        logging.info(
            "  Continuing training from checkpoint, will skip to saved global_step")
        logging.info("  Continuing training from epoch %d", epochs_trained)
        logging.info("  Continuing training from global step %d", global_step)
        logging.info(
            "  Will skip the first %d steps in the first epoch",
            steps_trained_in_current_epoch)

    tr_loss = 0.0
    model.zero_grad()
    set_seed(seed, n_gpu=n_gpu)  # Added here for reproductibility

    train_iterator = trange(epochs_trained,
                            int(num_train_epochs),
                            desc="Epoch",
                            disable=disable)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=disable)

        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
            # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            if model_type != "distilbert":
                inputs["token_type_ids"] = (batch[2] if model_type in [
                    "bert", "xlnet", "albert"] else None)
            outputs = model(**inputs)
            loss = outputs[0]
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            training_logs["loss"].append(loss.item())
            training_logs["learning_rate"].append(scheduler.get_last_lr()[0])
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if local_rank in [-1,
                              0] and save_steps > 0 and global_step % save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(
                    base_output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # Take care of distributed/parallel training
                model_to_save = (
                    model.module if hasattr(
                        model, "module") else model)
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(
                    hyperparams,
                    os.path.join(
                        output_dir,
                        "training_hyperparams.bin"))
                logging.info("Saving model checkpoint to %s", output_dir)
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(
                        output_dir,
                        "optimizer.pt"))
                torch.save(
                    scheduler.state_dict(),
                    os.path.join(
                        output_dir,
                        "scheduler.pt"))
                logging.info(
                    "Saving optimizer and scheduler states to %s",
                    output_dir)

                train_loss, train_results = evaluate(
                    train_dataset_to_eval, hyperparams, model)
                train_acc = (train_results.prediction ==
                             train_results.label).mean()
                dev_loss, dev_results = evaluate(
                    dev_dataset_to_eval, hyperparams, model)
                dev_acc = (dev_results.prediction == dev_results.label).mean()

                eval_logs["step"].append(global_step)
                eval_logs["train_acc"].append(train_acc)
                eval_logs["dev_acc"].append(dev_acc)
                eval_logs["train_loss"].append(train_loss)
                eval_logs["dev_loss"].append(dev_loss)

            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break
        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break

    training_logs = pd.DataFrame(training_logs)
    training_logs.to_csv(train_log_path, index=False)
    eval_logs = pd.DataFrame(eval_logs)
    eval_logs.to_csv(eval_log_path, index=False)

    return global_step, tr_loss / global_step
