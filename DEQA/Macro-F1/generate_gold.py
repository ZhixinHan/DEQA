import json


def main():
    with open("Twitter2015MASCDecisionDataset.json", "r") as file:
        gold_data = json.load(file)

    with open("gold.json", "w") as file:
        json.dump(gold_data["test"][2], file)


if __name__ == "__main__":
    main()