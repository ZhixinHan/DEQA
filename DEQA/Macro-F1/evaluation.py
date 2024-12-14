import json
from sklearn.metrics import precision_recall_fscore_support


def calculate_macro_f1(gold_data, predictions_data):
    gold_labels = []
    pred_labels = []

    for gold_sublist, pred_sublist in zip(gold_data, predictions_data):
        gold_dict = {item[0]: item[1] for item in gold_sublist}
        pred_dict = {item[0]: item[1] for item in pred_sublist}

        for identifier in gold_dict:
            if identifier in pred_dict:
                gold_labels.append(gold_dict[identifier])
                pred_labels.append(pred_dict[identifier])

    _, _, f1_score, _ = precision_recall_fscore_support(gold_labels, pred_labels, average='macro')
    return f1_score


def main():
    with open("gold.json", "r") as file:
        gold_data = json.load(file)

    with open("predictions.json", "r") as file:
        predictions_data = json.load(file)

    macro_f1 = calculate_macro_f1(gold_data, predictions_data)
    print("MASC Macro-average F1 Score:", macro_f1)


if __name__ == "__main__":
    main()
