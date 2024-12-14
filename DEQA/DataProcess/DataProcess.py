import os
from transformers import AutoTokenizer, AutoProcessor
import collections
import torch
import json
from tqdm import tqdm
from PIL import Image
import time
from LoadData import Twitter2015MASCDecisionLoad, \
    Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaLoad


class Twitter2015MASCDecisionProcessor:
    def __init__(self, release_or_debug, dataset_name):
        self.release_or_debug = release_or_debug
        self.dataset_name = dataset_name

        self.dataset_types = ['train', 'dev', 'test']
        self.dataset_folder = "datasets"
        self.text_dataset_folder = "textual"
        self.visual_dataset_folder = "visual"
        self.pre_processed = "pre_processed"
        self.data_dict = dict()
        self.text_pretrained_model_large = 'deberta-v3-large'
        self.text_pretrained_model_base = 'deberta-v3-large'
        self.CLIP = 'clip-vit-large-patch14-336'
        self.pretrained_model = "pretrained_model"
        self.padding_strategy = "max_length"
        self.sentence_and_question_max_length = 60
        self.pt_data = {"train": [], "dev": [], "test": []}
        self.dataset_name_pt = "Twitter2015MASCDecisionDataset"
        self.description = {}
        self.description_max_length = 470

        self.dataset_text_dir = os.path.join(self.dataset_folder, self.release_or_debug, self.dataset_name,
                                             self.text_dataset_folder)
        self.dataset_image_dir = os.path.join(self.dataset_folder, self.release_or_debug, self.dataset_name,
                                              self.visual_dataset_folder)
        self.output_path = os.path.join(self.pre_processed, self.dataset_name_pt)
        self.tokenizer_large = AutoTokenizer.from_pretrained(
            os.path.join(self.pretrained_model, self.text_pretrained_model_large), add_prefix_space=True)
        self.tokenizer_base = AutoTokenizer.from_pretrained(
            os.path.join(self.pretrained_model, self.text_pretrained_model_base), add_prefix_space=True)
        self.description_file_path = os.path.join(self.dataset_folder, self.release_or_debug, self.dataset_name,
                                                  "description_roberta.jsonl")
        self.processor = AutoProcessor.from_pretrained(os.path.join(self.pretrained_model, self.CLIP))

    def generate_data(self):
        self.get_description_dict()
        self.generate_sentence()
        self.generate_dataset()

    def generate_sentence(self):
        for dataset_type in self.dataset_types:
            data_file_name = dataset_type + ".txt"
            text_path = os.path.join(self.dataset_text_dir, data_file_name)
            sentence_dict = collections.defaultdict(list)
            sentence_list = []
            label_list = []
            pair_list = []
            image_list = []

            with open(text_path, 'r', encoding="utf-8") as file:

                while True:
                    text = file.readline().rstrip('\n').split()

                    if not text:
                        break

                    aspect_term = file.readline().rstrip('\n').split()
                    sentiment = file.readline().rstrip('\n')
                    image_name = file.readline().rstrip('\n')
                    start_pos = text.index("$T$")
                    end_pos = start_pos + len(aspect_term) - 1
                    text = text[:start_pos] + aspect_term + text[start_pos + 1:]
                    sentence_dict[" ".join(text)].append((start_pos, end_pos, int(sentiment), image_name))

                for key, value in sentence_dict.items():
                    text = key.split()
                    sentence_list.append(text)
                    num_key = len(text)
                    sentence_label = [0] * num_key
                    sentence_pair = []
                    image_list.append(value[0][3])

                    for value_in_value in value:
                        value_in_value_sentiment = value_in_value[2] + 1
                        sentence_label[value_in_value[0]] = value_in_value_sentiment + 2

                        for i in range(value_in_value[0] + 1, value_in_value[1] + 1):
                            sentence_label[i] = 1

                        sentence_pair.append(
                            (str(value_in_value[0]) + "-" + str(value_in_value[1]), value_in_value_sentiment))

                    label_list.append(sentence_label)
                    pair_list.append(sentence_pair)

                self.data_dict[dataset_type] = (sentence_list, label_list, pair_list, image_list)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        file_name = self.dataset_name_pt + ".json"
        file_path = os.path.join(self.output_path, file_name)

        with open(file_path, "w") as file:
            json.dump(self.data_dict, file)

    def generate_dataset(self):
        print(self.dataset_name_pt, ":")
        print("*" * 100)

        for dataset_type in self.dataset_types:
            sentence_list, label_list, pair_list, image_list = self.data_dict[dataset_type]
            print(dataset_type, "->", "generate_data")
            print("loading images:")
            self.wait_one_second()
            images = self.generate_image_list(image_list)
            print("processing data:")
            self.wait_one_second()
            progress_bar = self.generate_progress_bar(len(sentence_list))

            for text, label, pair, image, image_name in zip(sentence_list, label_list, pair_list, images, image_list):
                progress_bar.update(1)

                tokenized_input = self.tokenizer_large(text, return_tensors="pt", truncation=True,
                                                       is_split_into_words=True, padding=self.padding_strategy,
                                                       max_length=self.sentence_and_question_max_length)

                tokenized_description = self.tokenizer_base(self.description[image_name], return_tensors="pt",
                                                            truncation=True, padding=self.padding_strategy,
                                                            max_length=self.description_max_length)

                image_input = self.processor(images=image, return_tensors="pt")["pixel_values"]

                word_ids = tokenized_input.word_ids(batch_index=0)
                label_ids = []
                cross_label_ids = []
                label_length = len(label)
                pre_word_id = None

                for word_id in word_ids:

                    if word_id is None or word_id >= label_length:
                        label_ids.append(-100)
                        cross_label_ids.append(0)
                    else:
                        if pre_word_id != word_id:
                            label_ids.append(label[word_id])
                            cross_label_ids.append(label[word_id])
                        else:
                            label_ids.append(-100)
                            cross_label_ids.append(0)

                    pre_word_id = word_id

                tokenized_input["real_label"] = [pair]
                tokenized_input["cross_labels"] = torch.tensor([cross_label_ids])
                tokenized_input["labels"] = torch.tensor([label_ids])
                tokenized_input["description_input_ids"] = tokenized_description["input_ids"]
                tokenized_input["pixel_values"] = image_input
                tokenized_input["word_ids"] = torch.tensor(
                    [[-100 if i is None else i for i in word_ids]]
                )

                self.pt_data[dataset_type].append(tokenized_input)

            progress_bar.close()
            self.wait_one_second()
            self.save_pt(dataset_type)

    def generate_progress_bar(self, total):
        progress_bar = tqdm(
            total=total,
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            smoothing=0.1
        )
        return progress_bar

    def save_pt(self, dataset_type):
        file_name = self.dataset_name_pt + dataset_type.capitalize() + ".pt"
        file_path = os.path.join(self.output_path, file_name)
        print("saving......")
        torch.save(self.pt_data[dataset_type], file_path)
        print("finished!")

    def wait_one_second(self):
        start_time = time.perf_counter()

        while True:

            for i in range(10):
                continue

            current_time = time.perf_counter()
            if current_time - start_time >= 1:
                break

    def generate_image_list(self, image_list):
        images = []

        for image_name in tqdm(image_list):
            image_path = os.path.join(self.dataset_image_dir, image_name)
            image = Image.open(image_path)
            image = image.convert('RGB')
            images.append(image)

        return images

    def get_description_dict(self):
        with open(self.description_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                self.description[json_obj["image_name"]] = json_obj["description"]


class Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaProcessor:
    def __init__(self, release_or_debug, dataset_name):
        self.release_or_debug = release_or_debug
        self.dataset_name = dataset_name

        self.dataset_types = ['train', 'dev', 'test']
        self.dataset_folder = "datasets"
        self.text_dataset_folder = "textual"
        self.visual_dataset_folder = "visual"
        self.pre_processed = "pre_processed"
        self.data_dict = dict()
        self.text_pretrained_model_large = 'deberta-v3-large'
        self.text_pretrained_model_base = 'deberta-v3-base'
        self.CLIP = 'clip-vit-base-patch32'
        self.pretrained_model = "pretrained_model"
        self.padding_strategy = "max_length"
        self.sentence_and_question_max_length = 60
        self.pt_data = {"train": [], "dev": [], "test": []}
        self.dataset_name_pt = "Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaDataset"
        self.question = ["What", "aspect", "terms", "?"]
        self.description = {}
        self.description_max_length = 470

        self.dataset_text_dir = os.path.join(self.dataset_folder, self.release_or_debug, self.dataset_name,
                                             self.text_dataset_folder)
        self.dataset_image_dir = os.path.join(self.dataset_folder, self.release_or_debug, self.dataset_name,
                                              self.visual_dataset_folder)
        self.output_path = os.path.join(self.pre_processed, self.dataset_name_pt)
        self.tokenizer_large = AutoTokenizer.from_pretrained(
            os.path.join(self.pretrained_model, self.text_pretrained_model_large), add_prefix_space=True)
        self.tokenizer_base = AutoTokenizer.from_pretrained(
            os.path.join(self.pretrained_model, self.text_pretrained_model_base), add_prefix_space=True)
        self.description_file_path = os.path.join(self.dataset_folder, self.release_or_debug, self.dataset_name,
                                                  "description_roberta.jsonl")
        self.processor = AutoProcessor.from_pretrained(os.path.join(self.pretrained_model, self.CLIP))

    def generate_data(self):
        self.get_description_dict()
        self.generate_sentence()
        self.generate_dataset()

    def generate_sentence(self):
        for dataset_type in self.dataset_types:
            data_file_name = dataset_type + ".txt"
            text_path = os.path.join(self.dataset_text_dir, data_file_name)
            sentence_dict = collections.defaultdict(list)
            sentence_list = []
            label_list = []
            pair_list = []
            image_list = []

            with open(text_path, 'r', encoding="utf-8") as file:

                while True:
                    text = file.readline().rstrip('\n').split()

                    if not text:
                        break

                    aspect_term = file.readline().rstrip('\n').split()
                    _ = file.readline().rstrip('\n')
                    image_name = file.readline().rstrip('\n')
                    start_pos = text.index("$T$")
                    end_pos = start_pos + len(aspect_term) - 1
                    text = text[:start_pos] + aspect_term + text[start_pos + 1:]
                    sentence_dict[" ".join(text)].append((start_pos, end_pos, -1, image_name))

                for key, value in sentence_dict.items():
                    text = key.split()
                    sentence_list.append(text)
                    num_key = len(text)
                    sentence_label = [0] * num_key
                    sentence_pair = []
                    image_list.append(value[0][3])

                    for value_in_value in value:
                        value_in_value_sentiment = value_in_value[2] + 1
                        sentence_label[value_in_value[0]] = value_in_value_sentiment + 2

                        for i in range(value_in_value[0] + 1, value_in_value[1] + 1):
                            sentence_label[i] = 1

                        sentence_pair.append(
                            (str(value_in_value[0]) + "-" + str(value_in_value[1]), value_in_value_sentiment))

                    label_list.append(sentence_label)
                    pair_list.append(sentence_pair)

                self.data_dict[dataset_type] = (sentence_list, label_list, pair_list, image_list)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        file_name = self.dataset_name_pt + ".json"
        file_path = os.path.join(self.output_path, file_name)

        with open(file_path, "w") as file:
            json.dump(self.data_dict, file)

    def generate_dataset(self):
        print(self.dataset_name_pt, ":")
        print("*" * 100)

        for dataset_type in self.dataset_types:
            sentence_list, label_list, pair_list, image_list = self.data_dict[dataset_type]
            print(dataset_type, "->", "generate_data")
            print("loading images:")
            self.wait_one_second()
            images = self.generate_image_list(image_list)
            print("processing data:")
            self.wait_one_second()
            progress_bar = self.generate_progress_bar(len(sentence_list))

            for text, label, pair, image, image_name in zip(sentence_list, label_list, pair_list, images, image_list):
                progress_bar.update(1)

                tokenized_input = self.tokenizer_large(text, self.question, return_tensors="pt", truncation=True,
                                                       is_split_into_words=True, padding=self.padding_strategy,
                                                       max_length=self.sentence_and_question_max_length)

                tokenized_description = self.tokenizer_base(self.description[image_name], return_tensors="pt",
                                                            truncation=True, padding=self.padding_strategy,
                                                            max_length=self.description_max_length)

                image_input = self.processor(images=image, return_tensors="pt")["pixel_values"]

                word_ids = tokenized_input.word_ids(batch_index=0)
                word_ids = self.processing_word_ids(word_ids)
                label_ids = []
                cross_label_ids = []
                label_length = len(label)
                pre_word_id = None

                for word_id in word_ids:

                    if word_id is None or word_id >= label_length:
                        label_ids.append(-100)
                        cross_label_ids.append(0)
                    else:
                        if pre_word_id != word_id:
                            label_ids.append(label[word_id])
                            cross_label_ids.append(label[word_id])
                        else:
                            label_ids.append(-100)
                            cross_label_ids.append(0)

                    pre_word_id = word_id

                tokenized_input["real_label"] = [pair]
                tokenized_input["cross_labels"] = torch.tensor([cross_label_ids])
                tokenized_input["labels"] = torch.tensor([label_ids])
                tokenized_input["description_input_ids"] = tokenized_description["input_ids"]
                tokenized_input["pixel_values"] = image_input
                tokenized_input["word_ids"] = torch.tensor(
                    [[-100 if i is None else i for i in word_ids]]
                )

                self.pt_data[dataset_type].append(tokenized_input)

            progress_bar.close()
            self.wait_one_second()
            self.save_pt(dataset_type)

    def generate_progress_bar(self, total):
        progress_bar = tqdm(
            total=total,
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            smoothing=0.1
        )
        return progress_bar

    def save_pt(self, dataset_type):
        file_name = self.dataset_name_pt + dataset_type.capitalize() + ".pt"
        file_path = os.path.join(self.output_path, file_name)
        print("saving......")
        torch.save(self.pt_data[dataset_type], file_path)
        print("finished!")

    def processing_word_ids(self, word_ids):
        second_none_index = None
        none_count = 0

        for i, value in enumerate(word_ids):
            if value is None:
                none_count += 1
                if none_count == 2:
                    second_none_index = i
                    break

        if second_none_index is not None:
            for i in range(second_none_index + 1, len(word_ids)):
                word_ids[i] = None

        return word_ids

    def wait_one_second(self):
        start_time = time.perf_counter()

        while True:

            for i in range(10):
                continue

            current_time = time.perf_counter()
            if current_time - start_time >= 1:
                break

    def generate_image_list(self, image_list):
        images = []

        for image_name in tqdm(image_list):
            image_path = os.path.join(self.dataset_image_dir, image_name)
            image = Image.open(image_path)
            image = image.convert('RGB')
            images.append(image)

        return images

    def get_description_dict(self):
        with open(self.description_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                self.description[json_obj["image_name"]] = json_obj["description"]


class CombineProcessor:
    def __init__(self, release_or_debug, dataset_name):
        self.release_or_debug = release_or_debug
        self.dataset_name = dataset_name

        self.dataset_name_pt = "CombineDataset"
        self.dataset_types = ['train', 'dev', 'test']
        self.pre_processed = "pre_processed"
        self.pt_data = {"train": [], "dev": [], "test": []}

        self.output_path = os.path.join(self.pre_processed, self.dataset_name_pt)

        self.MATEprocessor = Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaProcessor(
            release_or_debug=self.release_or_debug, dataset_name=self.dataset_name)
        self.MASCprocessor = Twitter2015MASCDecisionProcessor(release_or_debug=self.release_or_debug, dataset_name=self.dataset_name)
        self.MASCload_data = Twitter2015MASCDecisionLoad()
        self.MATEload_data = Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaLoad()

    def generate_data(self):
        self.MATEprocessor.generate_data()
        self.MASCprocessor.generate_data()
        self.combine_data()

    def combine_data(self):
        print(self.dataset_name_pt, ":")
        print("*" * 100)

        for dataset_type in self.dataset_types:
            print(dataset_type, "->", "generate_data")
            print("loading data......")
            self.wait_one_second()
            MASCdata_inputs = self.MASCload_data.load_data(dataset_type=dataset_type)
            MATEdata_inputs = self.MATEload_data.load_data(dataset_type=dataset_type)
            print("finished!")
            print("combining data:")
            self.wait_one_second()
            progress_bar = self.generate_progress_bar(len(MASCdata_inputs))

            for MASCdata, MATEdata in zip(MASCdata_inputs, MATEdata_inputs):
                progress_bar.update(1)
                MASCdata["MATEinput_ids"] = MATEdata['input_ids']
                MASCdata["MATEtoken_type_ids"] = MATEdata['token_type_ids']
                MASCdata["MATEattention_mask"] = MATEdata['attention_mask']
                MASCdata["MATEreal_label"] = MATEdata['real_label']
                MASCdata["MATEcross_labels"] = MATEdata['cross_labels']
                MASCdata["MATElabels"] = MATEdata['labels']
                MASCdata["MATEdescription_input_ids"] = MATEdata['description_input_ids']
                MASCdata["MATEpixel_values"] = MATEdata['pixel_values']
                MASCdata["MATEword_ids"] = MATEdata['word_ids']

                self.pt_data[dataset_type].append(MASCdata)

            progress_bar.close()
            self.wait_one_second()
            self.save_pt(dataset_type)

    def wait_one_second(self):
        start_time = time.perf_counter()

        while True:

            for i in range(10):
                continue

            current_time = time.perf_counter()
            if current_time - start_time >= 1:
                break

    def generate_progress_bar(self, total):
        progress_bar = tqdm(
            total=total,
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            smoothing=0.1
        )
        return progress_bar

    def save_pt(self, dataset_type):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        file_name = self.dataset_name_pt + dataset_type.capitalize() + ".pt"
        file_path = os.path.join(self.output_path, file_name)
        print("saving......")
        torch.save(self.pt_data[dataset_type], file_path)
        print("finished!")
