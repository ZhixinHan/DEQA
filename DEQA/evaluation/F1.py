import json


def to_tuple(nested_list):
    for i in range(len(nested_list)):
        for j in range(len(nested_list[i])):
            nested_list[i][j] = tuple(nested_list[i][j])
    return nested_list


def cal_f1(p_pred_labels, p_pairs, is_result=True):
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


def main():
    with open("gold.json", "r") as file:
        gold = json.load(file)

    with open("predictions.json", "r") as file:
        predictions = json.load(file)

    to_tuple(gold)
    to_tuple(predictions)
    textPrecision, textRecall, textF1 = cal_f1(p_pred_labels=predictions, p_pairs=gold, is_result=False)
    print("MABSA:", "Precision:", textPrecision, "Recall:", textRecall, "F1:", textF1)


if __name__ == "__main__":
    main()
