import os
from transformers import EvalPrediction
from typing import Callable, Dict
import numpy as np
import json


class MASCMetrics:
    def __init__(self):
        pass

    def build_compute_metrics_function(self, inputs, real_label, MATEreal_label) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_function(p: EvalPrediction):
            text_pred_pairs = p.predictions[1]
            text_pred_pairs = self.return_raw_model_outputs(array_data=text_pred_pairs)
            text_pred_pairs = [[arr[arr != -200] for arr in sublist] for sublist in text_pred_pairs]
            logits = p.predictions[0]
            sentiments = np.argmax(logits, axis=-1)
            transformed_list = self.transform_list(text_pred_pairs, sentiments=sentiments)

            textPrecision, textRecall, textF1 = self.cal_f1(p_pred_labels=transformed_list, p_pairs=real_label,
                                                            is_result=False)

            torch_labels_cross_entropy_cat = p.predictions[2]
            acc = self.cal_acc(logits=logits, labels=torch_labels_cross_entropy_cat)

            return {"precision": textPrecision, "recall": textRecall, "f1": textF1, "acc": acc}

        return compute_metrics_function

    def cal_f1(self, p_pred_labels, p_pairs, is_result=True):
        gold_num = 0
        predict_num = 0
        correct_num = 0
        pred_pair_list = []
        for i, pred_label in enumerate(p_pred_labels):
            pred_pair = set(pred_label)
            true_pair = set(p_pairs[i])
            gold_num += len(true_pair)
            predict_num += len(list(pred_pair))
            pred_pair_list.append(pred_pair.copy())
            correct_num += len(true_pair & pred_pair)
        precision = 0
        recall = 0
        f1 = 0
        if predict_num != 0:
            precision = correct_num / predict_num
        if gold_num != 0:
            recall = correct_num / gold_num
        if precision != 0 or recall != 0:
            f1 = (2 * precision * recall) / (precision + recall)
        if is_result:
            return precision * 100, recall * 100, f1 * 100, pred_pair_list
        else:
            return precision * 100, recall * 100, f1 * 100

    def compute_test_metrics_function(self, p, inputs, real_label, path):
        text_pred_pairs = p.predictions[1]
        text_pred_pairs = self.return_raw_model_outputs(array_data=text_pred_pairs)
        text_pred_pairs = [[arr[arr != -200] for arr in sublist] for sublist in text_pred_pairs]
        logits = p.predictions[0]
        sentiments = np.argmax(logits, axis=-1)
        transformed_list = self.transform_list(text_pred_pairs, sentiments=sentiments)

        _, _, _, text_pred_pair_list = self.cal_f1(p_pred_labels=transformed_list, p_pairs=real_label,
                                                   is_result=True)

        pred_path = "predictions"
        pred_file = os.path.join(path, pred_path + ".json")

        if not os.path.exists(path):
            os.makedirs(path)

        converted_list = [list(s) for s in text_pred_pair_list]

        def convert_int64_to_int(item):
            if isinstance(item, list):
                return [convert_int64_to_int(subitem) for subitem in item]
            elif isinstance(item, tuple):
                return tuple(convert_int64_to_int(subitem) for subitem in item)
            elif isinstance(item, np.int64):
                return int(item)
            else:
                return item

        converted_list = convert_int64_to_int(converted_list)

        with open(pred_file, "w") as file:
            json.dump(converted_list, file)

        return converted_list

    def transform_list(self, lst, sentiments):
        result = []
        sentiment_index = 0

        for sub_list in lst:
            transformed = []
            for inner_list in sub_list:
                if len(inner_list) > 1:
                    min_val = min(inner_list)
                    max_val = max(inner_list)
                    transformed.append((f'{min_val}-{max_val}', sentiments[sentiment_index]))
                elif len(inner_list) == 1:
                    val = inner_list[0]
                    transformed.append((f'{val}-{val}', sentiments[sentiment_index]))
                sentiment_index += 1

            result.append(transformed)

        return result

    def cal_acc(self, logits, labels):
        pred_labels = np.argmax(logits, axis=-1)
        bool_ndarray = np.where(pred_labels == labels, 1, 0)
        count_of_ones = np.sum(bool_ndarray)
        acc = count_of_ones / len(bool_ndarray) * 100
        return acc

    def return_raw_model_outputs(self, array_data):
        original_list = []

        for subarray in array_data:
            if np.all(subarray == -200):
                original_list.append([])
            else:
                non_padding_rows = [row for row in subarray if not np.all(row == -200)]
                original_list.append(non_padding_rows)

        return original_list


class Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaMetrics:
    def __init__(self):
        pass

    def build_compute_metrics_function(self, inputs, real_label, MATEreal_label) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_function(p: EvalPrediction):
            text_pred_pairs = p.predictions
            text_pred_pairs = self.return_raw_model_outputs(array_data=text_pred_pairs)
            text_pred_pairs = [[arr[arr != -200] for arr in sublist] for sublist in text_pred_pairs]
            transformed_list = self.transform_list(text_pred_pairs)

            textPrecision, textRecall, textF1 = self.cal_f1(p_pred_labels=transformed_list, p_pairs=MATEreal_label,
                                                            is_result=False)

            return {"precision": textPrecision, "recall": textRecall, "f1": textF1}

        return compute_metrics_function

    def cal_f1(self, p_pred_labels, p_pairs, is_result=True):
        gold_num = 0
        predict_num = 0
        correct_num = 0
        pred_pair_list = []
        for i, pred_label in enumerate(p_pred_labels):
            pred_pair = set(pred_label)
            true_pair = set(p_pairs[i])
            gold_num += len(true_pair)
            predict_num += len(list(pred_pair))
            pred_pair_list.append(pred_pair.copy())
            correct_num += len(true_pair & pred_pair)
        precision = 0
        recall = 0
        f1 = 0
        if predict_num != 0:
            precision = correct_num / predict_num
        if gold_num != 0:
            recall = correct_num / gold_num
        if precision != 0 or recall != 0:
            f1 = (2 * precision * recall) / (precision + recall)
        if is_result:
            return precision * 100, recall * 100, f1 * 100, pred_pair_list
        else:
            return precision * 100, recall * 100, f1 * 100

    def compute_test_metrics_function(self, p, inputs, real_label, path):
        text_pred_pairs = p.predictions
        text_pred_pairs = self.return_raw_model_outputs(array_data=text_pred_pairs)
        text_pred_pairs = [[arr[arr != -200] for arr in sublist] for sublist in text_pred_pairs]
        transformed_list = self.transform_list(text_pred_pairs)

        _, _, _, text_pred_pair_list = self.cal_f1(p_pred_labels=transformed_list, p_pairs=real_label,
                                                   is_result=True)

        pred_path = "predictions"
        pred_file = os.path.join(path, pred_path + ".json")

        if not os.path.exists(path):
            os.makedirs(path)

        converted_list = [list(s) for s in text_pred_pair_list]

        def convert_int64_to_int(item):
            if isinstance(item, list):
                return [convert_int64_to_int(subitem) for subitem in item]
            elif isinstance(item, tuple):
                return tuple(convert_int64_to_int(subitem) for subitem in item)
            elif isinstance(item, np.int64):
                return int(item)
            else:
                return item

        converted_list = convert_int64_to_int(converted_list)

        with open(pred_file, "w") as file:
            json.dump(converted_list, file)

        return converted_list

    def transform_list(self, lst):
        result = []

        for sub_list in lst:
            transformed = []
            for inner_list in sub_list:
                if len(inner_list) > 1:
                    min_val = min(inner_list)
                    max_val = max(inner_list)
                    transformed.append((f'{min_val}-{max_val}', 0))
                elif len(inner_list) == 1:
                    val = inner_list[0]
                    transformed.append((f'{val}-{val}', 0))

            result.append(transformed)

        return result

    def return_raw_model_outputs(self, array_data):
        original_list = []

        for subarray in array_data:
            if np.all(subarray == -200):
                original_list.append([])
            else:
                non_padding_rows = [row for row in subarray if not np.all(row == -200)]
                original_list.append(non_padding_rows)

        return original_list
