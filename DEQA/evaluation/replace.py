import json


def convert_to_dict_list(nested_list):
    dict_list = []
    for sublist in nested_list:
        dict_item = {item[0]: item[1] for item in sublist}
        dict_list.append(dict_item)
    return dict_list


def main():
    with open("MATE - 1.json", "r") as file:
        MATE = json.load(file)

    with open("MASC.json", "r") as file:
        MASC = json.load(file)

    MATE_dict = convert_to_dict_list(MATE)
    MASC_dict = convert_to_dict_list(MASC)

    for mate, masc in zip(MATE_dict, MASC_dict):
        for key, value in masc.items():
            if key in mate:
                mate[key] = value

    converted_data = [list(d.items()) for d in MATE_dict]

    with open('predictions.json', 'w') as f:
        json.dump(converted_data, f)


if __name__ == "__main__":
    main()
