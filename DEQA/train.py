import argparse
import os
import torch
import random
import numpy as np
from SimpleDataset import SimpleDataset
from models.models import Twitter2015MASCDecisionModel, \
    Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaModel, \
    Twitter2015MASCDeBERTa_large_target_Model, Twitter2015MASCDescriptionDeBERTa_large_target_Model, \
    Twitter2015MASCCLIP_large_336_target_DeBERTaModel
from datetime import datetime
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from LOG import Logger, log_json
from LoadData import CombineLoad
from build_compute_metrics_function import MASCMetrics, \
    Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaMetrics
import json
import shutil


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--model_name', type=str,
                        default="Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaModel",
                        choices=["Twitter2015MASCDecisionModel",
                                 "Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaModel",
                                 "Twitter2015MASCDeBERTa_large_target_Model",
                                 "Twitter2015MASCDescriptionDeBERTa_large_target_Model",
                                 "Twitter2015MASCCLIP_large_336_target_DeBERTaModel"])
    parser.add_argument('--lr_scheduler_type', type=str, default="linear", choices=["linear"])
    parser.add_argument('--Is_save_model_history', type=str, default="True", choices=["True", "False"])
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--early_stopping_threshold', type=float, default=0.01)
    parser.add_argument('--metric_for_best_model', type=str, default="f1", choices=["f1"])
    parser.add_argument('--greater_is_better', type=str, default="True", choices=["True", "False"])
    parser.add_argument('--Is_not_train', type=str, default="False", choices=["True", "False"])
    parser.add_argument('--save_total_limit', default=3, type=int)
    parser.add_argument('--Is_save_model', type=str, default="False", choices=["True", "False"])
    parser.add_argument('--Is_save_optimizer', type=str, default="False", choices=["True", "False"])

    args = parser.parse_args()
    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    global model, load_data, compute_metrics
    args = get_parameters()
    seed = args.seed
    batch_size = args.batch_size
    num_train_epochs = args.num_train_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    gradient_accumulation_steps = args.gradient_accumulation_steps
    warmup_ratio = args.warmup_ratio
    model_name = args.model_name
    lr_scheduler_type = args.lr_scheduler_type
    Is_save_model_history = args.Is_save_model_history.lower() == 'true'
    early_stopping_patience = args.early_stopping_patience
    early_stopping_threshold = args.early_stopping_threshold
    metric_for_best_model = args.metric_for_best_model
    greater_is_better = args.greater_is_better.lower() == 'true'
    Is_not_train = args.Is_not_train.lower() == 'true'
    save_total_limit = args.save_total_limit
    Is_save_model = args.Is_save_model.lower() == 'true'
    Is_save_optimizer = args.Is_save_optimizer.lower() == 'true'

    output_dir = 'output_dir'
    trained_model = "trained_model"
    save_args = "save_args"
    result_file_name = "result"

    set_random_seed(seed)

    match model_name:
        case "Twitter2015MASCDecisionModel":
            model = Twitter2015MASCDecisionModel()
            load_data = CombineLoad(batch_size=batch_size)
            compute_metrics = MASCMetrics()
        case "Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaModel":
            model = Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaModel()
            load_data = CombineLoad(batch_size=batch_size)
            compute_metrics = Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaMetrics()
        case "Twitter2015MASCDeBERTa_large_target_Model":
            model = Twitter2015MASCDeBERTa_large_target_Model()
            load_data = CombineLoad(batch_size=batch_size)
            compute_metrics = MASCMetrics()
        case "Twitter2015MASCDescriptionDeBERTa_large_target_Model":
            model = Twitter2015MASCDescriptionDeBERTa_large_target_Model()
            load_data = CombineLoad(batch_size=batch_size)
            compute_metrics = MASCMetrics()
            model.load_state_dict(torch.load(
                os.path.join("trained_model", "Twitter2015MASCDeBERTa_large_target_Model", "pytorch_model.bin")))
        case "Twitter2015MASCCLIP_large_336_target_DeBERTaModel":
            model = Twitter2015MASCCLIP_large_336_target_DeBERTaModel()
            load_data = CombineLoad(batch_size=batch_size)
            compute_metrics = MASCMetrics()

    print("loading data......")
    train_data_inputs, train_real_label, MATEtrain_real_label = load_data.load_data(dataset_type="train")
    dev_data_inputs, dev_real_label, MATEdev_real_label = load_data.load_data(dataset_type="dev")
    print("finished!")

    train_dataset = SimpleDataset(inputs=train_data_inputs)
    dev_dataset = SimpleDataset(inputs=dev_data_inputs)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d+%H_%M_%S")

    output_path = os.path.join(output_dir, formatted_time)

    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        load_best_model_at_end=True,
        save_total_limit=save_total_limit,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better
    )

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=dev_dataset,
                      compute_metrics=compute_metrics.build_compute_metrics_function(inputs=dev_data_inputs,
                                                                                     real_label=dev_real_label,
                                                                                     MATEreal_label=MATEdev_real_label),
                      callbacks=[Logger(),
                                 EarlyStoppingCallback(early_stopping_patience=early_stopping_patience,
                                                       early_stopping_threshold=early_stopping_threshold)]
                      )

    print("model.state_dict().keys():")

    for key in model.state_dict().keys():
        state_dict = key + ": " + str(model.state_dict()[key].size())
        print(state_dict)

        with open(os.path.join(output_path, "state_dict.txt"), "a") as file:
            file.write(state_dict + "\n")

    if not Is_not_train:
        trainer.train()

    log_file = os.path.join(output_path, "logs.json")

    with open(log_file, "w") as file:
        json.dump(log_json, file, indent=4)

    trained_model_path = os.path.join(trained_model, model_name)

    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)

    if Is_save_model:
        trainer.save_model(trained_model_path)

    with open(os.path.join(trained_model_path, "time.txt"), "w") as file:
        file.write(formatted_time)

    if not Is_save_model_history:
        for item in os.listdir(output_path):
            if item.startswith('checkpoint-'):
                folder_path = os.path.join(output_path, item)
                os.remove(os.path.join(folder_path, "pytorch_model.bin"))
                os.remove(os.path.join(folder_path, "optimizer.pt"))

                shutil.copy(os.path.join(folder_path, "trainer_state.json"),
                            os.path.join(trained_model_path, "trainer_state.json"))
    else:
        if not Is_save_optimizer:
            for item in os.listdir(output_path):
                if item.startswith('checkpoint-'):
                    folder_path = os.path.join(output_path, item)
                    os.remove(os.path.join(folder_path, "optimizer.pt"))

    save_args_path = os.path.join(output_path, save_args + ".json")

    with open(save_args_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    with open(os.path.join(trained_model_path, save_args + ".json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    result_file_path = os.path.join(trained_model_path, result_file_name + ".json")

    if os.path.exists(result_file_path):
        os.remove(result_file_path)


if __name__ == '__main__':
    main()
