import argparse
from DataProcess.DataProcess import CombineProcessor


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release_or_debug', type=str, default="debug", choices=["release", "debug"])
    parser.add_argument('--processor_name', type=str, default="CombineProcessor", choices=["CombineProcessor"])
    parser.add_argument('--dataset_name', type=str, default="twitter2015", choices=["twitter2015", "twitter2017"])

    args = parser.parse_args()
    return args


def main():
    global processor
    args = get_parameters()
    release_or_debug = args.release_or_debug
    processor_name = args.processor_name
    dataset_name = args.dataset_name

    match processor_name:
        case "CombineProcessor":
            processor = CombineProcessor(release_or_debug=release_or_debug, dataset_name=dataset_name)

    processor.generate_data()


if __name__ == '__main__':
    main()
