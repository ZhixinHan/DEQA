import os
import torch
import copy


class Twitter2015MASCDecisionLoad:
    def __init__(self):
        self.pre_processed = "pre_processed"
        self.dataset_name_pt = "Twitter2015MASCDecisionDataset"

    def load_data(self, dataset_type):
        filename = self.dataset_name_pt + dataset_type.capitalize() + ".pt"
        data_input_file = os.path.join(self.pre_processed, self.dataset_name_pt, filename)
        data_inputs = torch.load(data_input_file)

        return data_inputs


class Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaLoad:
    def __init__(self):
        self.pre_processed = "pre_processed"
        self.dataset_name_pt = "Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaDataset"

    def load_data(self, dataset_type):
        filename = self.dataset_name_pt + dataset_type.capitalize() + ".pt"
        data_input_file = os.path.join(self.pre_processed, self.dataset_name_pt, filename)
        data_inputs = torch.load(data_input_file)

        return data_inputs


class CombineLoad:
    def __init__(self, batch_size):
        self.pre_processed = "pre_processed"
        self.dataset_name_pt = "CombineDataset"
        self.batch_size = batch_size

    def load_data(self, dataset_type):
        filename = self.dataset_name_pt + dataset_type.capitalize() + ".pt"
        data_input_file = os.path.join(self.pre_processed, self.dataset_name_pt, filename)
        data_inputs = torch.load(data_input_file)

        remainder = len(data_inputs) % self.batch_size

        if remainder != 0:
            num_to_add = self.batch_size - remainder
            data_inputs.extend([copy.deepcopy(data_inputs[-1]) for _ in range(num_to_add)])

        pairs = []

        for i in data_inputs:
            pairs.append(i["real_label"][0])
            i.pop("real_label")

        MATEpairs = []

        for i in data_inputs:
            MATEpairs.append(i["MATEreal_label"][0])
            i.pop("MATEreal_label")

        return data_inputs, pairs, MATEpairs
