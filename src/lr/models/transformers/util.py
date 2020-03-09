from transformers.data.processors.utils import InputExample, InputFeatures
from transformers.data.processors.utils import DataProcessor
from torch.utils.data import TensorDataset, DataLoader
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


logging.basicConfig(filename='example.log', level=logging.INFO)
spaces = re.compile(' +')


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def remove_first_space(x):
    """
    remove_first_space from word x

    :param x: word
    :type x: str
    :return: word withou space in front
    :rtype: str
    """
    try:
        if x[0] == " ":
            return x[1:]
        else:
            return x
    except IndexError:
        return x


def simple_pre_process_text_df(data, text_column):
    """
    preprocess all input text from dataframe by
    lowering, removing non words, removing
    space in the first position and
    removing double spaces


    :param data: data frame with the colum 'text'
    :type data: pd.DataFrame
    :param text_column: colum text_column
    :type text_column: str
    """

    data.loc[:, text_column] = data.loc[:,
                                        text_column].apply(lambda x: x.lower())
    data.loc[:, text_column] = data.loc[:, text_column].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))  # noqa
    data.loc[:, text_column] = data.loc[:, text_column].apply(remove_first_space)  # noqa remove space in the first position
    data.loc[:, text_column] = data.loc[:, text_column].apply((lambda x: spaces.sub(" ", x)))  # noqa remove double spaces


def pre_process_nli_df(data):
    """
    Apply preprocess on the input text from a NLI dataframe

    :param data: data frame with the colum 'text'
    :type data: pd.DataFrame
    """
    simple_pre_process_text_df(data, text_column="premise")
    simple_pre_process_text_df(data, text_column="hypothesis")


def filter_df_by_label(df, drop_label='-'):
    """
    drop observations with label 'drop_label'
    """
    return df.loc[df.label != drop_label]


class NLIProcessor(DataProcessor):
    """Processor for the any nli dataf frame in csv
       (columns = premise | hypothesis | label)"""

    def read_and_clean_csv(self, path):
        df = pd.read_csv(path)
        df = filter_df_by_label(df.dropna()).reset_index(drop=True)
        pre_process_nli_df(df)
        return df

    def df2examples(self, df, set_type):
        df = filter_df_by_label(df.dropna()).reset_index(drop=True)
        pre_process_nli_df(df)
        examples = self._create_examples(df, set_type)
        return examples

    def get_train_examples(self, path):
        logging.info("creating train examples for {}".format(path))
        return self._create_examples(self.read_and_clean_csv(path), "train")

    def get_dev_examples(self, path):
        logging.info("creating dev examples for {}".format(path))
        return self._create_examples(self.read_and_clean_csv(path), "dev")

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def get_label_map(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        return label_map

    def _create_examples(self, df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        n = df.shape[0]
        for i in range(n):
            example = df.loc[i]
            guid = "{}-{}".format(set_type, example.name)
            input_example = InputExample(guid=guid,
                                         text_a=example.premise,
                                         text_b=example.hypothesis,
                                         label=example.label)

            examples.append(input_example)
        return examples


def convert_examples_to_features(examples,
                                 tokenizer,
                                 label_map,
                                 max_length=512,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = len(examples)
        if ex_index % 100 == 0:
            logging.info("converting example %d/%d" % (ex_index, len_examples))
        inputs = tokenizer.encode_plus(example.text_a,
                                       example.text_b,
                                       add_special_tokens=True,
                                       max_length=max_length)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1]
                              * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] *
                              padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + \
                ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + \
                ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length)
        label = label_map[example.label]
        features.append(InputFeatures(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      label=label))
    return features


def features2dataset(cached_features_file):
    assert os.path.exists(cached_features_file)
    logging.info("Loading features from cached file %s", cached_features_file)
    features = torch.load(cached_features_file)

#     torch.distributed.barrier()

    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_labels)
    return dataset


def load_and_cache_examples(hyperparams, tokenizer, evaluate=False):
    overwrite_cache = hyperparams["overwrite_cache"]
    max_seq_length = hyperparams["max_seq_length"]
    local_rank = hyperparams["local_rank"]
    cached_path = hyperparams["cached_path"]
    train_path = hyperparams["train_path"]
    dev_path = hyperparams["dev_path"]

    processor = NLIProcessor()
    label_map = processor.get_label_map()

    if local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    mode = "dev" if evaluate else "train"
    cached_features_file = cached_path + \
        "cached_{}_{}".format(mode, max_seq_length)

    if os.path.exists(cached_features_file) and not overwrite_cache:
        logging.info(
            "Loading features from cached file %s",
            cached_features_file)
        features = torch.load(cached_features_file)

    else:
        logging.info(
            "Creating features from dataset file at %s",
            cached_features_file)
        if evaluate:
            examples = processor.get_dev_examples(dev_path)
        else:
            examples = processor.get_train_examples(train_path)

        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_map=label_map,
                                                max_length=max_seq_length)
        if local_rank in [-1, 0]:
            logging.info(
                "Saving features into cached file %s",
                cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset,
    # and the others will use the cache
    if local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_labels)
    return dataset


def train(train_dataset, model, tokenizer, hyperparams):

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

    output_dir = hyperparams["output_dir"]
    log_path = os.path.join(output_dir, "log.csv")
    fp16_opt_level = hyperparams["fp16_opt_level"]
    fp16 = hyperparams["fp16"]

    model_name_or_path = hyperparams["model_name_or_path"]
    opt_path = os.path.join(model_name_or_path, "optimizer.pt")
    sche_path = os.path.join(model_name_or_path, "scheduler.pt")

    training_logs = {"loss": [],
                     "learning_rate": []}
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
                    output_dir, "checkpoint-{}".format(global_step))
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

            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break
        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break

    training_logs = pd.DataFrame(training_logs)
    training_logs.to_csv(log_path, index=False)
    return global_step, tr_loss / global_step


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
