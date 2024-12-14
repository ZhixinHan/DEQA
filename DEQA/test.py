import argparse
import os
import json
from models.models import Twitter2015MASCDecisionModel, \
    Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaModel
from LoadData import CombineLoad
from build_compute_metrics_function import MASCMetrics, \
    Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaMetrics
from SimpleDataset import SimpleDataset
import torch
from transformers import TrainingArguments, Trainer


def get_parameters(defaults=None):
    if defaults is None:
        defaults = {'batch_size': 1}

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=defaults['batch_size'])
    parser.add_argument('--model_name', type=str,
                        default="Twitter2015MASCDecisionModel",
                        choices=["Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaModel",
                                 "Twitter2015MASCDecisionModel"])

    args = parser.parse_args()
    return args


def main():
    global model, load_data, compute_metrics
    args = get_parameters()
    model_name = args.model_name

    trained_model = "trained_model"
    output_dir = "output_dir"
    save_args = "save_args"
    test_folder_name = 'test'
    result_file_name = "result"

    trained_model_path = os.path.join(trained_model, model_name)

    with open(os.path.join(trained_model_path, "time.txt"), "r") as file:
        formatted_time = file.readline()

    output_path = os.path.join(output_dir, formatted_time)
    test_folder_path = os.path.join(output_path, test_folder_name)

    with open(os.path.join(trained_model_path, save_args + ".json"), 'r') as file:
        defaults = json.load(file)

    args = get_parameters(defaults)
    batch_size = args.batch_size

    match model_name:
        case "Twitter2015MASCDecisionModel":
            model = Twitter2015MASCDecisionModel()
            load_data = CombineLoad(batch_size=batch_size)
            compute_metrics = MASCMetrics()
        case "Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaModel":
            model = Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaModel()
            load_data = CombineLoad(batch_size=batch_size)
            compute_metrics = Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaMetrics()

    model.load_state_dict(torch.load(os.path.join(trained_model_path, "pytorch_model.bin")))

    print("loading data......")
    test_data_inputs, test_real_label, MATEtest_real_label = load_data.load_data(dataset_type="test")
    print("finished!")

    test_dataset = SimpleDataset(inputs=test_data_inputs)

    training_args = TrainingArguments(
        output_dir=test_folder_path,
        per_device_eval_batch_size=batch_size
    )

    trainer = Trainer(model=model,
                      args=training_args,
                      compute_metrics=compute_metrics.build_compute_metrics_function(inputs=test_data_inputs,
                                                                                     real_label=test_real_label,
                                                                                     MATEreal_label=MATEtest_real_label)
                      )

    output = trainer.predict(test_dataset=test_dataset)
    print(output.metrics)
    converted_list = compute_metrics.compute_test_metrics_function(p=output, inputs=test_data_inputs, real_label=test_real_label,
                                                  path=test_folder_path)

    log_file = os.path.join(test_folder_path, result_file_name + ".json")
    log_json = {"test": output.metrics}

    with open(log_file, "w") as file:
        json.dump(log_json, file, indent=4)

    with open(os.path.join(trained_model_path, result_file_name + ".json"), "w") as file:
        json.dump(log_json, file, indent=4)

    with open(os.path.join(trained_model_path, "predictions.json"), "w") as file:
        json.dump(converted_list, file)


if __name__ == '__main__':
    main()
