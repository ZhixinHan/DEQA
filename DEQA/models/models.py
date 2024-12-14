from torch import nn
import os
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoProcessor, CLIPVisionModel, \
    CLIPTextModelWithProjection
import copy
import torch
import torch.nn.functional as F
from torchcrf import CRF


class Twitter2015MASCDeBERTa_large_target_Model_ensemble(nn.Module):
    def __init__(self):
        super(Twitter2015MASCDeBERTa_large_target_Model_ensemble, self).__init__()
        self.pretrained_model = "pretrained_model"
        self.padding_strategy = "max_length"
        self.question = ["What", "is", "the", "sentiment", "polarity", "of", "the", "<target>", "</target>", "?"]
        self.aspect_term_max_length = 60
        self.aspect_term_num = 60
        self.text_pretrained_model_large = 'deberta-v3-large'
        self.question_max_length = 80
        self.special_tokens_dict = {'additional_special_tokens': ['<target>', '</target>']}

        self.pretrained_model_path_large = os.path.join(self.pretrained_model, self.text_pretrained_model_large)
        self.config_roberta = AutoConfig.from_pretrained(self.pretrained_model_path_large)
        self.aspect_start_index = self.question.index("<target>") + 1

        self.tokenizer_large = AutoTokenizer.from_pretrained(self.pretrained_model_path_large, add_prefix_space=True)
        self.tokenizer_large.add_special_tokens(self.special_tokens_dict)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(self.config_roberta.hidden_size, 3)
        self.dropout = nn.Dropout(self.config_roberta.hidden_dropout_prob)
        self.RoBERTa_large = AutoModel.from_pretrained(self.pretrained_model_path_large)
        self.RoBERTa_large.resize_token_embeddings(len(self.tokenizer_large))

    def forward(self, input_ids, gold_aspects, gold_pair_list_index_batch, gold_sentiments, **kwargs):
        sentence = []

        for i in input_ids:
            sentence.append(self.tokenizer_large.tokenize(self.tokenizer_large.decode(i, skip_special_tokens=True)))

        sentence_merge = self.merge_tokens_two(nested_list=copy.deepcopy(sentence))
        batch_size = input_ids.shape[0]
        logits_batch = []
        torch_labels_cross_entropy = []

        for i in range(batch_size):
            all_aspects_in_a_sentence = []
            for j in gold_aspects[i]:
                all_aspects_in_a_sentence.append(j)

            question_input = []
            sentence_input = []
            count_list = self.generate_count_list(lst=all_aspects_in_a_sentence)

            for aspect, count in zip(all_aspects_in_a_sentence, count_list):
                question_input.append(
                    self.question[:self.aspect_start_index] + aspect + self.question[self.aspect_start_index:])
                sentence_input.append(
                    self.insert_tags_around_sequence(lst=sentence_merge[i].copy(), sequence=aspect, occurrence=count))

            tokenized_sentence = self.tokenizer_large(question_input, sentence_input,
                                                      return_tensors="pt", truncation=True,
                                                      padding=self.padding_strategy,
                                                      max_length=self.question_max_length, is_split_into_words=True)

            sentence_output = self.RoBERTa_large(input_ids=tokenized_sentence["input_ids"].cuda(),
                                                 attention_mask=tokenized_sentence["attention_mask"].cuda(),
                                                 return_dict=self.config_roberta.use_return_dict)

            logits = self.classifier(sentence_output["last_hidden_state"][:, 8, :])
            logits_dropout = self.dropout(logits)
            logits_batch.append(logits_dropout)
            torch_labels = torch.tensor(gold_sentiments[i])
            torch_labels_cross_entropy.append(torch_labels)

        logits_batch_cat = torch.cat(logits_batch, dim=0)
        torch_labels_cross_entropy_cat = torch.cat(torch_labels_cross_entropy, dim=0).cuda()
        loss = self.cross_entropy_loss(logits_batch_cat, torch_labels_cross_entropy_cat)

        output_dict = {"loss": loss, "logits": logits_batch_cat,
                       "gold_pair_list_index_batch": gold_pair_list_index_batch,
                       "torch_labels_cross_entropy_cat": torch_labels_cross_entropy_cat}

        self.to_tensor(output_dict)
        self.padding_and_truncation(output_tensor=output_dict)

        return output_dict

    def to_tensor(self, output_dict):
        for i, item in enumerate(output_dict["gold_pair_list_index_batch"]):
            for j, inner_item in enumerate(item):
                output_dict["gold_pair_list_index_batch"][i][j] = torch.tensor(inner_item).clone().detach()

    def padding_and_truncation(self, output_tensor):
        gold_pair_list_index_batch = output_tensor["gold_pair_list_index_batch"]

        big_tensor_two_format = self.get_big_tensor_two_format(tensor_groups=gold_pair_list_index_batch,
                                                               max_length=self.aspect_term_max_length,
                                                               max_num=self.aspect_term_num)

        output_tensor["gold_pair_list_index_batch"] = big_tensor_two_format

    def merge_tokens_two(self, nested_list):
        for sub_list in nested_list:
            i = 1
            while i < len(sub_list):
                if not sub_list[i].startswith("▁"):
                    sub_list[i - 1] += sub_list[i]
                    sub_list.pop(i)
                else:
                    i += 1

        result_list = [[s[1:] if s.startswith('▁') else s for s in sublist] for sublist in nested_list]

        return result_list

    def get_big_tensor_two_format(self, tensor_groups, max_length, max_num):
        processed_group_tensors = []

        for group in tensor_groups:
            new_tensor = torch.full((max_num, max_length), -200)

            for index, tensor in enumerate(group):
                if index < max_num:
                    if tensor.size(0) < max_length:
                        padding_size = max_length - tensor.size(0)
                        padded_tensor = F.pad(tensor, (0, padding_size), value=-200)
                    elif tensor.size(0) > max_length:
                        padded_tensor = tensor[:max_length]
                    else:
                        padded_tensor = tensor

                    new_tensor[index] = padded_tensor

            processed_group_tensors.append(new_tensor)

        big_tensor_group = torch.stack(processed_group_tensors)
        return big_tensor_group

    def generate_count_list(self, lst):
        count_dict = {}
        result = []
        for item in lst:
            element = tuple(item)
            if element in count_dict:
                count_dict[element] += 1
            else:
                count_dict[element] = 1
            result.append(count_dict[element])
        return result

    def insert_tags_around_sequence(self, lst, sequence, occurrence):
        length = len(sequence)
        count = 0

        i = 0
        while i <= len(lst) - length:
            if lst[i:i + length] == sequence:
                count += 1
                if count == occurrence:
                    lst.insert(i, '<target>')
                    lst.insert(i + length + 1, '</target>')
                    break
            i += 1

        return lst


class Twitter2015MASCDescriptionDeBERTa_large_target_Model_ensemble(nn.Module):
    def __init__(self):
        super(Twitter2015MASCDescriptionDeBERTa_large_target_Model_ensemble, self).__init__()
        self.pretrained_model = "pretrained_model"
        self.padding_strategy = "max_length"
        self.question = "What is the sentiment polarity of the <target></target> in the sentence \"\"?"
        self.aspect_term_max_length = 60
        self.aspect_term_num = 60
        self.text_pretrained_model_large = 'deberta-v3-large'
        self.question_max_length = 512
        self.special_tokens_dict = {'additional_special_tokens': ['<target>', '</target>']}

        self.pretrained_model_path_large = os.path.join(self.pretrained_model, self.text_pretrained_model_large)
        self.config_roberta = AutoConfig.from_pretrained(self.pretrained_model_path_large)
        self.aspect_start_index = self.question.index(">") + 1

        self.tokenizer_large = AutoTokenizer.from_pretrained(self.pretrained_model_path_large, add_prefix_space=True)
        self.tokenizer_large.add_special_tokens(self.special_tokens_dict)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(self.config_roberta.hidden_size, 3)
        self.dropout = nn.Dropout(self.config_roberta.hidden_dropout_prob)
        self.RoBERTa_large = AutoModel.from_pretrained(self.pretrained_model_path_large)
        self.RoBERTa_large.resize_token_embeddings(len(self.tokenizer_large))

    def forward(self, input_ids, description_input_ids, gold_aspects, gold_pair_list_index_batch, gold_sentiments,
                **kwargs):
        sentence = []
        description_sentence = []

        for i in description_input_ids:
            description_sentence.append(self.tokenizer_large.decode(i, skip_special_tokens=True))

        for i in input_ids:
            sentence.append(
                self.tokenizer_large.decode(i, skip_special_tokens=True).strip())

        batch_size = input_ids.shape[0]
        logits_batch = []
        torch_labels_cross_entropy = []

        for i in range(batch_size):
            all_aspects_in_a_sentence = []
            for j in gold_aspects[i]:
                all_aspects_in_a_sentence.append(j)

            question_input = []
            count_list = self.generate_count_list(lst=all_aspects_in_a_sentence)

            for aspect, count in zip(all_aspects_in_a_sentence, count_list):
                join_aspect = ' '.join(aspect)
                front = self.question[:self.aspect_start_index] + " " + join_aspect + " " + self.question[
                                                                                            self.aspect_start_index:]

                sentence_start_index = front.rfind('"')

                modified_string = self.replace_nth_occurrence(s=sentence[i], target=join_aspect,
                                                              replacement='<target> ' + join_aspect + ' </target>',
                                                              n=count)

                behind = front[:sentence_start_index] + modified_string + front[sentence_start_index:]
                question_input.append(behind)

            tokenized_sentence = self.tokenizer_large(question_input, [description_sentence[i]] * len(gold_aspects[i]),
                                                      return_tensors="pt", truncation=True,
                                                      padding=self.padding_strategy,
                                                      max_length=self.question_max_length)

            sentence_output = self.RoBERTa_large(input_ids=tokenized_sentence["input_ids"].cuda(),
                                                 attention_mask=tokenized_sentence["attention_mask"].cuda(),
                                                 return_dict=self.config_roberta.use_return_dict)

            logits = self.classifier(sentence_output["last_hidden_state"][:, 8, :])
            logits_dropout = self.dropout(logits)
            logits_batch.append(logits_dropout)
            torch_labels = torch.tensor(gold_sentiments[i])
            torch_labels_cross_entropy.append(torch_labels)

        logits_batch_cat = torch.cat(logits_batch, dim=0)
        torch_labels_cross_entropy_cat = torch.cat(torch_labels_cross_entropy, dim=0).cuda()
        loss = self.cross_entropy_loss(logits_batch_cat, torch_labels_cross_entropy_cat)

        output_dict = {"loss": loss, "logits": logits_batch_cat,
                       "gold_pair_list_index_batch": gold_pair_list_index_batch,
                       "torch_labels_cross_entropy_cat": torch_labels_cross_entropy_cat}

        self.to_tensor(output_dict)
        self.padding_and_truncation(output_tensor=output_dict)

        return output_dict

    def to_tensor(self, output_dict):
        for i, item in enumerate(output_dict["gold_pair_list_index_batch"]):
            for j, inner_item in enumerate(item):
                output_dict["gold_pair_list_index_batch"][i][j] = inner_item.clone().detach()

    def padding_and_truncation(self, output_tensor):
        gold_pair_list_index_batch = output_tensor["gold_pair_list_index_batch"]

        big_tensor_two_format = self.get_big_tensor_two_format(tensor_groups=gold_pair_list_index_batch,
                                                               max_length=self.aspect_term_max_length,
                                                               max_num=self.aspect_term_num)

        output_tensor["gold_pair_list_index_batch"] = big_tensor_two_format

    def get_big_tensor_two_format(self, tensor_groups, max_length, max_num):
        processed_group_tensors = []

        for group in tensor_groups:
            new_tensor = torch.full((max_num, max_length), -200)

            for index, tensor in enumerate(group):
                if index < max_num:
                    if tensor.size(0) < max_length:
                        padding_size = max_length - tensor.size(0)
                        padded_tensor = F.pad(tensor, (0, padding_size), value=-200)
                    elif tensor.size(0) > max_length:
                        padded_tensor = tensor[:max_length]
                    else:
                        padded_tensor = tensor

                    new_tensor[index] = padded_tensor

            processed_group_tensors.append(new_tensor)

        big_tensor_group = torch.stack(processed_group_tensors)
        return big_tensor_group

    def generate_count_list(self, lst):
        count_dict = {}
        result = []
        for item in lst:
            element = tuple(item)
            if element in count_dict:
                count_dict[element] += 1
            else:
                count_dict[element] = 1
            result.append(count_dict[element])
        return result

    def replace_nth_occurrence(self, s, target, replacement, n):
        index = -1
        for _ in range(n):
            index = s.find(target, index + 1)
            if index == -1:
                return s
        return s[:index] + replacement + s[index + len(target):]


class Twitter2015MASCCLIP_large_336_target_DeBERTaModel_ensemble(nn.Module):
    class MFBFusion_outer(nn.Module):
        def __init__(self, input_dim1, input_dim2, hidden_dim, R):
            super().__init__()
            self.input_dim1 = input_dim1
            self.input_dim2 = input_dim2
            self.hidden_dim = hidden_dim
            self.R = R
            self.linear1 = nn.Linear(input_dim1, hidden_dim * R)
            self.linear2 = nn.Linear(input_dim2, hidden_dim * R)

        def forward(self, inputs1, inputs2):
            num_region = 1
            if inputs1.dim() == 3:
                num_region = inputs1.size(1)
            h1 = self.linear1(inputs1)
            h2 = self.linear2(inputs2)
            z = h1 * h2
            z = z.view(z.size(0), num_region, self.hidden_dim, self.R)
            z = z.sum(3).squeeze(1)
            return z

    class MultiHeadATTN(nn.Module):
        class MFBFusion_inner(nn.Module):
            def __init__(self, input_dim1, input_dim2, hidden_dim, R):
                super().__init__()
                self.input_dim1 = input_dim1
                self.input_dim2 = input_dim2
                self.hidden_dim = hidden_dim
                self.R = R
                self.linear1 = nn.Linear(input_dim1, hidden_dim * R)
                self.linear2 = nn.Linear(input_dim2, hidden_dim * R)

            def forward(self, inputs1, inputs2):
                num_region = 1
                if inputs1.dim() == 3:
                    num_region = inputs1.size(1)
                h1 = self.linear1(inputs1)
                h2 = self.linear2(inputs2)
                z = h1 * h2
                z = z.view(z.size(0), num_region, self.hidden_dim, self.R)
                z = z.sum(3).squeeze(1)
                return z

        def __init__(self, query_dim, kv_dim, mfb_input_dim, mfb_hidden_dim, num_head, att_dim):
            super().__init__()
            assert att_dim % num_head == 0
            self.num_head = num_head
            self.att_dim = att_dim
            self.R = 1

            self.attn_w_1_q = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(query_dim, mfb_input_dim),
                nn.ReLU()
            )

            self.attn_w_1_k = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(kv_dim, mfb_input_dim),
                nn.ReLU()
            )

            self.attn_score_fusion = self.MFBFusion_inner(mfb_input_dim, mfb_input_dim, mfb_hidden_dim, self.R)

            self.attn_score_mapping = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(mfb_hidden_dim, num_head)
            )

            self.softmax = nn.Softmax(dim=1)

            self.align_q = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(kv_dim, round(att_dim / num_head)),
                        nn.Tanh()
                    ) for _ in range(num_head)
                ]
            )

        def forward(self, query, key_value):
            num_region = key_value.shape[1]
            q = self.attn_w_1_q(query).unsqueeze(1).repeat(1, num_region, 1)
            k = self.attn_w_1_k(key_value)
            alphas = self.attn_score_fusion(q, k)
            alphas = self.attn_score_mapping(alphas)
            alphas = self.softmax(alphas)
            output = torch.bmm(alphas.transpose(1, 2), key_value)
            list_v = [e.squeeze(dim=1) for e in torch.split(output, 1, dim=1)]
            alpha = torch.split(alphas, 1, dim=2)
            align_feat = [self.align_q[head_id](x_v) for head_id, x_v in enumerate(list_v)]
            align_feat = torch.cat(align_feat, 1)
            return align_feat, alpha

    def __init__(self):
        super().__init__()
        self.pretrained_model = "pretrained_model"
        self.question_max_length = 77
        self.padding_strategy = "max_length"
        self.question = "What is the sentiment polarity of the <target></target> in the sentence \"\"?"
        self.aspect_term_max_length = 60
        self.aspect_term_num = 60
        self.text_pretrained_model_large = 'deberta-v3-large'
        self.CLIP = 'clip-vit-large-patch14-336'
        self.R = 8

        self.pretrained_model_path_large = os.path.join(self.pretrained_model, self.text_pretrained_model_large)
        self.aspect_start_index = self.question.index(">") + 1
        self.pretrained_model_path_clip = os.path.join(self.pretrained_model, self.CLIP)
        self.config_clip = AutoConfig.from_pretrained(self.pretrained_model_path_clip)
        self.attn_mfb_input_dim = round(
            self.config_clip.projection_dim / self.config_clip.text_config.num_attention_heads)
        self.attn_mfb_hidden_dim = self.attn_mfb_input_dim
        self.attn_output_dim = self.config_clip.text_config.num_attention_heads * self.attn_mfb_input_dim
        self.fusion_q_feature_dim = self.attn_mfb_input_dim
        self.fusion_mfb_hidden_dim = self.attn_mfb_hidden_dim

        self.tokenizer_large = AutoTokenizer.from_pretrained(self.pretrained_model_path_large, add_prefix_space=True)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.processor = AutoProcessor.from_pretrained(self.pretrained_model_path_clip)
        self.CLIPVisionEncoder = CLIPVisionModel.from_pretrained(self.pretrained_model_path_clip)
        self.CLIPTextEncoderWithProjection = CLIPTextModelWithProjection.from_pretrained(
            self.pretrained_model_path_clip)
        self.classifier_linear = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.fusion_mfb_hidden_dim, 3))
        self.fusion = self.MFBFusion_outer(input_dim1=self.attn_output_dim,
                                           input_dim2=self.fusion_q_feature_dim,
                                           hidden_dim=self.fusion_mfb_hidden_dim, R=self.R)
        self.attn = self.MultiHeadATTN(query_dim=self.config_clip.projection_dim,
                                       kv_dim=self.config_clip.vision_config.hidden_size,
                                       mfb_input_dim=self.attn_mfb_input_dim, mfb_hidden_dim=self.attn_mfb_hidden_dim,
                                       num_head=self.config_clip.text_config.num_attention_heads,
                                       att_dim=self.attn_output_dim)
        self.q_feature_linear = nn.Sequential(nn.Dropout(0.5),
                                              nn.Linear(self.config_clip.projection_dim,
                                                        self.fusion_q_feature_dim), nn.ReLU())

    def forward(self, input_ids, pixel_values, gold_aspects, gold_pair_list_index_batch, gold_sentiments, **kwargs):
        sentence = []

        for i in input_ids:
            sentence.append(
                self.tokenizer_large.decode(i, skip_special_tokens=True).strip())

        batch_size = input_ids.shape[0]
        logits_batch = []
        torch_labels_cross_entropy = []

        for i in range(batch_size):
            all_aspects_in_a_sentence = []
            for j in gold_aspects[i]:
                all_aspects_in_a_sentence.append(j)

            question_input = []
            count_list = self.generate_count_list(lst=all_aspects_in_a_sentence)

            for aspect, count in zip(all_aspects_in_a_sentence, count_list):
                join_aspect = ' '.join(aspect)
                front = self.question[:self.aspect_start_index] + " " + join_aspect + " " + self.question[
                                                                                            self.aspect_start_index:]

                sentence_start_index = front.rfind('"')

                modified_string = self.replace_nth_occurrence(s=sentence[i], target=join_aspect,
                                                              replacement='<target> ' + join_aspect + ' </target>',
                                                              n=count)

                behind = front[:sentence_start_index] + modified_string + front[sentence_start_index:]
                question_input.append(behind)

            tokenized_question = self.processor(text=question_input, return_tensors="pt", truncation=True,
                                                padding=self.padding_strategy, max_length=self.question_max_length)

            vision_output = self.CLIPVisionEncoder(
                pixel_values=pixel_values[i].unsqueeze(0).repeat(len(gold_aspects[i]), 1, 1, 1),
                return_dict=self.config_clip.use_return_dict)

            text_output = self.CLIPTextEncoderWithProjection(input_ids=tokenized_question["input_ids"].cuda(),
                                                             attention_mask=tokenized_question["attention_mask"].cuda(),
                                                             return_dict=self.config_clip.use_return_dict)

            align_q_feature, _ = self.attn(text_output["text_embeds"], vision_output["last_hidden_state"].detach())
            original_q_feature = self.q_feature_linear(text_output["text_embeds"])
            x = self.fusion(align_q_feature, original_q_feature)
            logits = self.classifier_linear(x)
            logits_batch.append(logits)
            torch_labels = torch.tensor(gold_sentiments[i])
            torch_labels_cross_entropy.append(torch_labels)

        logits_batch_cat = torch.cat(logits_batch, dim=0)
        torch_labels_cross_entropy_cat = torch.cat(torch_labels_cross_entropy, dim=0).cuda()
        loss = self.cross_entropy_loss(logits_batch_cat, torch_labels_cross_entropy_cat)

        output_dict = {"loss": loss, "logits": logits_batch_cat,
                       "gold_pair_list_index_batch": gold_pair_list_index_batch,
                       "torch_labels_cross_entropy_cat": torch_labels_cross_entropy_cat}

        self.to_tensor(output_dict)
        self.padding_and_truncation(output_tensor=output_dict)

        return output_dict

    def to_tensor(self, output_dict):
        for i, item in enumerate(output_dict["gold_pair_list_index_batch"]):
            for j, inner_item in enumerate(item):
                output_dict["gold_pair_list_index_batch"][i][j] = inner_item.clone().detach()

    def padding_and_truncation(self, output_tensor):
        gold_pair_list_index_batch = output_tensor["gold_pair_list_index_batch"]

        big_tensor_two_format = self.get_big_tensor_two_format(tensor_groups=gold_pair_list_index_batch,
                                                               max_length=self.aspect_term_max_length,
                                                               max_num=self.aspect_term_num)

        output_tensor["gold_pair_list_index_batch"] = big_tensor_two_format

    def get_big_tensor_two_format(self, tensor_groups, max_length, max_num):
        processed_group_tensors = []

        for group in tensor_groups:
            new_tensor = torch.full((max_num, max_length), -200)

            for index, tensor in enumerate(group):
                if index < max_num:
                    if tensor.size(0) < max_length:
                        padding_size = max_length - tensor.size(0)
                        padded_tensor = F.pad(tensor, (0, padding_size), value=-200)
                    elif tensor.size(0) > max_length:
                        padded_tensor = tensor[:max_length]
                    else:
                        padded_tensor = tensor

                    new_tensor[index] = padded_tensor

            processed_group_tensors.append(new_tensor)

        big_tensor_group = torch.stack(processed_group_tensors)
        return big_tensor_group

    def generate_count_list(self, lst):
        count_dict = {}
        result = []
        for item in lst:
            element = tuple(item)
            if element in count_dict:
                count_dict[element] += 1
            else:
                count_dict[element] = 1
            result.append(count_dict[element])
        return result

    def replace_nth_occurrence(self, s, target, replacement, n):
        index = -1
        for _ in range(n):
            index = s.find(target, index + 1)
            if index == -1:
                return s
        return s[:index] + replacement + s[index + len(target):]


class Twitter2015MASCDecisionModel(nn.Module):
    def __init__(self):
        super(Twitter2015MASCDecisionModel, self).__init__()
        self.trained_model = "trained_model"
        self.pretrained_model = "pretrained_model"
        self.text_pretrained_model_large = 'deberta-v3-large'

        self.pretrained_model_path_large = os.path.join(self.pretrained_model, self.text_pretrained_model_large)

        self.tokenizer_large = AutoTokenizer.from_pretrained(self.pretrained_model_path_large, add_prefix_space=True)

        self.trained_model_path__Twitter2015MASCDeBERTa_large_target_Model = os.path.join(self.trained_model,
                                                                                          "Twitter2015MASCDeBERTa_large_target_Model")
        self.trained_model_path__Twitter2015MASCDescriptionDeBERTa_large_target_Model = os.path.join(self.trained_model,
                                                                                                     "Twitter2015MASCDescriptionDeBERTa_large_target_Model")
        self.trained_model_path__Twitter2015MASCCLIP_large_336_target_DeBERTaModel = os.path.join(self.trained_model,
                                                                                                  "Twitter2015MASCCLIP_large_336_target_DeBERTaModel")

        self.FreezeTwitter2015MASCDeBERTa_large_target_Model = Twitter2015MASCDeBERTa_large_target_Model_ensemble()
        self.FreezeTwitter2015MASCDescriptionDeBERTa_large_target_Model = Twitter2015MASCDescriptionDeBERTa_large_target_Model_ensemble()
        self.FreezeTwitter2015MASCCLIP_large_336_target_DeBERTaModel = Twitter2015MASCCLIP_large_336_target_DeBERTaModel_ensemble()

        self.FreezeTwitter2015MASCDeBERTa_large_target_Model.load_state_dict(
            torch.load(
                os.path.join(self.trained_model_path__Twitter2015MASCDeBERTa_large_target_Model, "pytorch_model.bin")))
        self.FreezeTwitter2015MASCDescriptionDeBERTa_large_target_Model.load_state_dict(
            torch.load(os.path.join(self.trained_model_path__Twitter2015MASCDescriptionDeBERTa_large_target_Model,
                                    "pytorch_model.bin")))
        self.FreezeTwitter2015MASCCLIP_large_336_target_DeBERTaModel.load_state_dict(
            torch.load(os.path.join(self.trained_model_path__Twitter2015MASCCLIP_large_336_target_DeBERTaModel,
                                    "pytorch_model.bin")))

        self.FreezeTwitter2015MASCDeBERTa_large_target_Model.eval()
        self.FreezeTwitter2015MASCDescriptionDeBERTa_large_target_Model.eval()
        self.FreezeTwitter2015MASCCLIP_large_336_target_DeBERTaModel.eval()

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, cross_labels, word_ids, description_input_ids, pixel_values, **kwargs):
        gold_aspects, gold_pair_list_index_batch, gold_sentiments = self.get_aspects(input_ids=input_ids,
                                                                                     word_ids=word_ids,
                                                                                     text_pred_labels=cross_labels)

        with torch.no_grad():
            output = self.FreezeTwitter2015MASCDeBERTa_large_target_Model(input_ids=input_ids,
                                                                          gold_aspects=gold_aspects,
                                                                          gold_pair_list_index_batch=gold_pair_list_index_batch,
                                                                          gold_sentiments=gold_sentiments)

            output_Description = self.FreezeTwitter2015MASCDescriptionDeBERTa_large_target_Model(input_ids=input_ids,
                                                                                                 description_input_ids=description_input_ids,
                                                                                                 gold_aspects=gold_aspects,
                                                                                                 gold_pair_list_index_batch=gold_pair_list_index_batch,
                                                                                                 gold_sentiments=gold_sentiments)

            output_CLIP = self.FreezeTwitter2015MASCCLIP_large_336_target_DeBERTaModel(input_ids=input_ids,
                                                                                       pixel_values=pixel_values,
                                                                                       gold_aspects=gold_aspects,
                                                                                       gold_pair_list_index_batch=gold_pair_list_index_batch,
                                                                                       gold_sentiments=gold_sentiments)

        final_logits = F.softmax(output["logits"], dim=1) + F.softmax(output_Description["logits"], dim=1) + F.softmax(
            output_CLIP["logits"], dim=1)
        loss = self.cross_entropy_loss(final_logits, output["torch_labels_cross_entropy_cat"])

        output_dict = {"loss": loss, "logits": final_logits,
                       "gold_pair_list_index_batch": output["gold_pair_list_index_batch"],
                       "torch_labels_cross_entropy_cat": output["torch_labels_cross_entropy_cat"]}

        return output_dict

    def get_aspects(self, input_ids, word_ids, text_pred_labels):
        pred_pair_list = self.get_predicted_pair(p_pred_labels=text_pred_labels, word_ids=word_ids)
        pred_pair_list_index_batch = []
        pred_pair_list_index_single = []

        for i in pred_pair_list:
            for j in i:
                pred_pair_list_index_single.append(list(range(j[0], j[1] + 1)) + [j[2]])
            pred_pair_list_index_batch.append(pred_pair_list_index_single)
            pred_pair_list_index_single = []

        pred_pair_list_index_batch, gold_sentiments = self.separate_tensors(data=pred_pair_list_index_batch)

        original_sentence = []

        for i in input_ids:
            original_sentence.append(self.tokenizer_large.tokenize(self.tokenizer_large.decode(i)))

        positions_single = []
        positions_batch = []

        for index, spans in enumerate(pred_pair_list_index_batch):
            for single_span in spans:
                positions_single.append(
                    [i for i, value in enumerate(word_ids[index]) if value in single_span])
            positions_batch.append(positions_single)
            positions_single = []

        aspect_term = []
        selected_data = []

        for positions, sentence in zip(positions_batch, original_sentence):
            for position in positions:
                selected_data.append([sentence[i] for i in position])
            aspect_term.append(selected_data)
            selected_data = []

        aspects = self.merge_tokens_three(nested_list=copy.deepcopy(aspect_term))

        return aspects, pred_pair_list_index_batch, gold_sentiments

    def get_predicted_pair(self, p_pred_labels, word_ids):
        pred_pair_list = []
        for i, pred_label in enumerate(p_pred_labels):
            word_ids_in_word_ids = word_ids[i]
            flag = False
            pred_pair = set()
            sentiment = 0
            start_pos = 0
            end_pos = 0
            for j, pp in enumerate(pred_label):
                if word_ids_in_word_ids[j] == -100:
                    if flag:
                        pred_pair.add((start_pos, end_pos,
                                       sentiment))
                        flag = False
                    continue
                if word_ids_in_word_ids[j] != word_ids_in_word_ids[
                    j - 1]:
                    if pp > 1:
                        if flag:
                            pred_pair.add((start_pos, end_pos,
                                           sentiment))
                        start_pos = word_ids_in_word_ids[j]
                        end_pos = word_ids_in_word_ids[j]
                        sentiment = pp - 2
                        flag = True
                    elif pp == 1:
                        if flag:
                            end_pos = word_ids_in_word_ids[
                                j]
                    else:
                        if flag:
                            pred_pair.add((start_pos, end_pos,
                                           sentiment))
                        flag = False
            pred_pair_list.append(pred_pair.copy())

        return pred_pair_list

    def separate_tensors(self, data):
        values_without_tensors = []
        tensors = []

        for sublist in data:
            current_sublist = []
            current_tensors = []
            for item in sublist:
                current_sublist.append(item[:-1])
                current_tensors.append(item[-1])
            values_without_tensors.append(current_sublist)
            tensors.append(current_tensors)

        return values_without_tensors, tensors

    def merge_tokens_three(self, nested_list):
        for sub_list in nested_list:
            for token_list in sub_list:
                i = 1
                while i < len(token_list):
                    if not token_list[i].startswith("▁"):
                        token_list[i - 1] += token_list[i]
                        token_list.pop(i)
                    else:
                        i += 1

        result_list = [[[s[1:] if s.startswith('▁') else s for s in sublist] for sublist in subsublist] for subsublist
                       in nested_list]
        return result_list


class Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaModel(nn.Module):
    class Twitter2015MATEQuestionRoBERTa_large_DropoutCRFModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_labels = 3
            self.text_pretrained_model = 'deberta-v3-large'
            self.pretrained_model = "pretrained_model"

            self.pretrained_model_path = os.path.join(self.pretrained_model, self.text_pretrained_model)
            self.config_roberta = AutoConfig.from_pretrained(self.pretrained_model_path)

            self.roberta = AutoModel.from_pretrained(self.pretrained_model_path)
            self.classifier = nn.Linear(self.config_roberta.hidden_size, self.num_labels)
            self.CRF = CRF(self.num_labels, batch_first=True)
            self.dropout = nn.Dropout(self.config_roberta.hidden_dropout_prob)

        def forward(self, input_ids, attention_mask, labels, cross_labels):
            text_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask,
                                        return_dict=self.config_roberta.use_return_dict)

            text_last_hidden_state = text_outputs["last_hidden_state"]
            text_logits = self.classifier(text_last_hidden_state)
            logits_dropout = self.dropout(text_logits)
            mask = (labels != -100)
            mask[:, 0] = 1
            text_loss = -self.CRF(logits_dropout, cross_labels, mask=mask) / 10
            return {"loss": text_loss, "logits": logits_dropout}

    class Twitter2015MATEDescription_pooler_output_DropoutCrossEntropyLossQuestionRoBERTa_large_DropoutCRFModel(
        nn.Module):
        def __init__(self):
            super().__init__()
            self.text_pretrained_model = 'deberta-v3-base'
            self.pretrained_model = "pretrained_model"
            self.padding_strategy = "max_length"
            self.description_max_length = 512
            self.question = "Is <target></target> the aspect term of the sentence \"\"?"
            self.aspect_term_max_length = 60
            self.aspect_term_num = 60
            self.text_pretrained_model_large = 'deberta-v3-large'
            self.special_tokens_dict = {'additional_special_tokens': ['<target>', '</target>']}

            self.pretrained_model_path = os.path.join(self.pretrained_model, self.text_pretrained_model)
            self.pretrained_model_path_large = os.path.join(self.pretrained_model, self.text_pretrained_model_large)
            self.config_roberta = AutoConfig.from_pretrained(self.pretrained_model_path)
            self.aspect_start_index = self.question.index(">") + 1

            self.roberta_description = AutoModel.from_pretrained(self.pretrained_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_path, add_prefix_space=True)
            self.tokenizer.add_special_tokens(self.special_tokens_dict)
            self.roberta_description.resize_token_embeddings(len(self.tokenizer))
            self.tokenizer_large = AutoTokenizer.from_pretrained(self.pretrained_model_path_large,
                                                                 add_prefix_space=True)
            self.classifier_yes_or_no = nn.Linear(self.config_roberta.hidden_size, 2)
            self.cross_entropy_loss = nn.CrossEntropyLoss()
            self.description_dropout = nn.Dropout(self.config_roberta.hidden_dropout_prob)

        def forward(self, input_ids, description_input_ids, output, pred_aspects, pred_pair_list_index_batch,
                    gold_aspects, gold_pair_list_index_batch):
            yes_or_no_labels = self.generate_labels(pred_aspects=pred_pair_list_index_batch,
                                                    gold_aspects=gold_pair_list_index_batch)

            yes_or_no_labels = self.add_value_to_empty_sublists(lst=yes_or_no_labels, value=-100)
            description_sentence = []
            sentence = []

            for i in description_input_ids:
                description_sentence.append(self.tokenizer.decode(i, skip_special_tokens=True))

            for i in input_ids:
                sentence.append(
                    self.tokenizer_large.decode(i, skip_special_tokens=True).replace('What aspect terms?', '').strip())

            batch_size = description_input_ids.shape[0]
            final_predicted_result_batch = []
            final_predicted_result_pair_batch = []
            logits_yes_or_no_batch = []
            pred_labels_yes_or_no_batch = []
            yes_or_no_torch_labels_cross_entropy = []

            for i in range(batch_size):
                all_aspects_in_a_sentence = []
                for j in pred_aspects[i]:
                    all_aspects_in_a_sentence.append(j)

                question_input = []
                count_list = self.generate_count_list(lst=all_aspects_in_a_sentence)

                for aspect, count in zip(all_aspects_in_a_sentence, count_list):
                    join_aspect = ' '.join(aspect)
                    front = self.question[:self.aspect_start_index] + " " + join_aspect + " " + self.question[
                                                                                                self.aspect_start_index:]

                    sentence_start_index = front.rfind('"')

                    modified_string = self.replace_nth_occurrence(s=sentence[i], target=join_aspect,
                                                                  replacement='<target> ' + join_aspect + ' </target>',
                                                                  n=count)

                    behind = front[:sentence_start_index] + modified_string + front[sentence_start_index:]
                    question_input.append(behind)

                try:
                    tokenized_description = self.tokenizer(question_input,
                                                           [description_sentence[i]] * len(pred_aspects[i]),
                                                           return_tensors="pt", truncation=True,
                                                           padding=self.padding_strategy,
                                                           max_length=self.description_max_length)
                except IndexError:
                    tokenized_description = self.tokenizer(self.question, description_sentence[i], return_tensors="pt",
                                                           truncation=True,
                                                           padding=self.padding_strategy,
                                                           max_length=self.description_max_length)

                description_output = self.roberta_description(input_ids=tokenized_description["input_ids"].cuda(),
                                                              attention_mask=tokenized_description[
                                                                  "attention_mask"].cuda(),
                                                              return_dict=self.config_roberta.use_return_dict)

                logits_yes_or_no = self.classifier_yes_or_no(description_output["last_hidden_state"][:, 3, :])
                logits_yes_or_no_dropout = self.description_dropout(logits_yes_or_no)
                yes_or_no_torch_labels = torch.tensor(yes_or_no_labels[i])
                yes_or_no_torch_labels_cross_entropy.append(yes_or_no_torch_labels)
                pred_labels_yes_or_no = torch.argmax(logits_yes_or_no_dropout, dim=-1)

                final_predicted_result = []
                final_predicted_result_pair = []

                for i_n_d_e_x, yes_or_no in enumerate(pred_labels_yes_or_no):
                    if yes_or_no == 1 and len(pred_aspects[i]) != 0:
                        final_predicted_result.append(pred_aspects[i][i_n_d_e_x])
                        final_predicted_result_pair.append(pred_pair_list_index_batch[i][i_n_d_e_x])

                final_predicted_result_batch.append(final_predicted_result)
                final_predicted_result_pair_batch.append(final_predicted_result_pair)
                logits_yes_or_no_batch.append(logits_yes_or_no_dropout)
                pred_labels_yes_or_no_batch.append(pred_labels_yes_or_no)

            logits_yes_or_no_batch_cat = torch.cat(logits_yes_or_no_batch, dim=0)
            yes_or_no_torch_labels_cross_entropy_cat = torch.cat(yes_or_no_torch_labels_cross_entropy, dim=0).cuda()
            description_loss = self.cross_entropy_loss(logits_yes_or_no_batch_cat,
                                                       yes_or_no_torch_labels_cross_entropy_cat)

            if torch.isnan(description_loss) == torch.tensor(True):
                description_loss = torch.tensor(0.0, requires_grad=True).cuda()

            output_dict = {"description_loss": description_loss, "logits_yes_or_no_batch_cat": logits_yes_or_no_batch_cat}

            return output_dict

        def generate_labels(self, pred_aspects, gold_aspects):
            labels = []

            for pred_list, gold_list in zip(pred_aspects, gold_aspects):
                label_list = []
                for pred_item in pred_list:
                    if pred_item in gold_list:
                        label_list.append(1)
                    else:
                        label_list.append(0)
                labels.append(label_list)

            return labels

        def add_value_to_empty_sublists(self, lst, value):
            for sublst in lst:
                if not sublst:
                    sublst.append(value)
            return lst

        def generate_count_list(self, lst):
            count_dict = {}
            result = []
            for item in lst:
                element = tuple(item)
                if element in count_dict:
                    count_dict[element] += 1
                else:
                    count_dict[element] = 1
                result.append(count_dict[element])
            return result

        def replace_nth_occurrence(self, s, target, replacement, n):
            index = -1
            for _ in range(n):
                index = s.find(target, index + 1)
                if index == -1:
                    return s
            return s[:index] + replacement + s[index + len(target):]

    class Twitter2015MATECLIPDropoutCrossEntropyLossQuestionRoBERTa_large_DropoutCRFModel(nn.Module):
        class MFBFusion_outer(nn.Module):
            def __init__(self, input_dim1, input_dim2, hidden_dim, R):
                super().__init__()
                self.input_dim1 = input_dim1
                self.input_dim2 = input_dim2
                self.hidden_dim = hidden_dim
                self.R = R
                self.linear1 = nn.Linear(input_dim1, hidden_dim * R)
                self.linear2 = nn.Linear(input_dim2, hidden_dim * R)

            def forward(self, inputs1, inputs2):
                num_region = 1
                if inputs1.dim() == 3:
                    num_region = inputs1.size(1)
                h1 = self.linear1(inputs1)
                h2 = self.linear2(inputs2)
                z = h1 * h2
                z = z.view(z.size(0), num_region, self.hidden_dim, self.R)
                z = z.sum(3).squeeze(1)
                return z

        class MultiHeadATTN(nn.Module):
            class MFBFusion_inner(nn.Module):
                def __init__(self, input_dim1, input_dim2, hidden_dim, R):
                    super().__init__()
                    self.input_dim1 = input_dim1
                    self.input_dim2 = input_dim2
                    self.hidden_dim = hidden_dim
                    self.R = R
                    self.linear1 = nn.Linear(input_dim1, hidden_dim * R)
                    self.linear2 = nn.Linear(input_dim2, hidden_dim * R)

                def forward(self, inputs1, inputs2):
                    num_region = 1
                    if inputs1.dim() == 3:
                        num_region = inputs1.size(1)
                    h1 = self.linear1(inputs1)
                    h2 = self.linear2(inputs2)
                    z = h1 * h2
                    z = z.view(z.size(0), num_region, self.hidden_dim, self.R)
                    z = z.sum(3).squeeze(1)
                    return z

            def __init__(self, query_dim, kv_dim, mfb_input_dim, mfb_hidden_dim, num_head, att_dim):
                super().__init__()
                assert att_dim % num_head == 0
                self.num_head = num_head
                self.att_dim = att_dim
                self.R = 1

                self.attn_w_1_q = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(query_dim, mfb_input_dim),
                    nn.ReLU()
                )

                self.attn_w_1_k = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(kv_dim, mfb_input_dim),
                    nn.ReLU()
                )

                self.attn_score_fusion = self.MFBFusion_inner(mfb_input_dim, mfb_input_dim, mfb_hidden_dim, self.R)

                self.attn_score_mapping = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(mfb_hidden_dim, num_head)
                )

                self.softmax = nn.Softmax(dim=1)

                self.align_q = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(kv_dim, round(att_dim / num_head)),
                            nn.Tanh()
                        ) for _ in range(num_head)
                    ]
                )

            def forward(self, query, key_value):
                num_region = key_value.shape[1]
                q = self.attn_w_1_q(query).unsqueeze(1).repeat(1, num_region, 1)
                k = self.attn_w_1_k(key_value)
                alphas = self.attn_score_fusion(q, k)
                alphas = self.attn_score_mapping(alphas)
                alphas = self.softmax(alphas)
                output = torch.bmm(alphas.transpose(1, 2), key_value)
                list_v = [e.squeeze(dim=1) for e in torch.split(output, 1, dim=1)]
                alpha = torch.split(alphas, 1, dim=2)
                align_feat = [self.align_q[head_id](x_v) for head_id, x_v in enumerate(list_v)]
                align_feat = torch.cat(align_feat, 1)
                return align_feat, alpha

        def __init__(self):
            super().__init__()
            self.pretrained_model = "pretrained_model"
            self.question_max_length = 77
            self.question = "Is <target></target> the aspect term of the sentence \"\"?"
            self.aspect_term_max_length = 60
            self.aspect_term_num = 60
            self.text_pretrained_model_large = 'deberta-v3-large'
            self.CLIP = 'clip-vit-base-patch32'
            self.padding_strategy = "max_length"
            self.R = 8

            self.pretrained_model_path_large = os.path.join(self.pretrained_model, self.text_pretrained_model_large)
            self.aspect_start_index = self.question.index(">") + 1
            self.pretrained_model_path_clip = os.path.join(self.pretrained_model, self.CLIP)
            self.config_clip = AutoConfig.from_pretrained(self.pretrained_model_path_clip)
            self.attn_mfb_input_dim = round(
                self.config_clip.projection_dim / self.config_clip.text_config.num_attention_heads)
            self.attn_mfb_hidden_dim = self.attn_mfb_input_dim
            self.attn_output_dim = self.config_clip.text_config.num_attention_heads * self.attn_mfb_input_dim
            self.fusion_q_feature_dim = self.attn_mfb_input_dim
            self.fusion_mfb_hidden_dim = self.attn_mfb_hidden_dim

            self.tokenizer_large = AutoTokenizer.from_pretrained(self.pretrained_model_path_large,
                                                                 add_prefix_space=True)
            self.cross_entropy_loss = nn.CrossEntropyLoss()
            self.processor = AutoProcessor.from_pretrained(self.pretrained_model_path_clip)
            self.CLIPVisionEncoder = CLIPVisionModel.from_pretrained(self.pretrained_model_path_clip)
            self.CLIPTextEncoderWithProjection = CLIPTextModelWithProjection.from_pretrained(
                self.pretrained_model_path_clip)
            self.classifier_linear = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.fusion_mfb_hidden_dim, 2))
            self.fusion = self.MFBFusion_outer(input_dim1=self.attn_output_dim,
                                               input_dim2=self.fusion_q_feature_dim,
                                               hidden_dim=self.fusion_mfb_hidden_dim, R=self.R)
            self.attn = self.MultiHeadATTN(query_dim=self.config_clip.projection_dim,
                                           kv_dim=self.config_clip.vision_config.hidden_size,
                                           mfb_input_dim=self.attn_mfb_input_dim,
                                           mfb_hidden_dim=self.attn_mfb_hidden_dim,
                                           num_head=self.config_clip.text_config.num_attention_heads,
                                           att_dim=self.attn_output_dim)
            self.q_feature_linear = nn.Sequential(nn.Dropout(0.5),
                                                  nn.Linear(self.config_clip.projection_dim,
                                                            self.fusion_q_feature_dim), nn.ReLU())

        def forward(self, input_ids, pixel_values, output, pred_aspects, pred_pair_list_index_batch, gold_aspects,
                    gold_pair_list_index_batch):
            yes_or_no_labels = self.generate_labels(pred_aspects=pred_pair_list_index_batch,
                                                    gold_aspects=gold_pair_list_index_batch)

            yes_or_no_labels = self.add_value_to_empty_sublists(lst=yes_or_no_labels, value=-100)
            batch_size = input_ids.shape[0]
            final_predicted_result_batch = []
            final_predicted_result_pair_batch = []
            logits_yes_or_no_batch = []
            pred_labels_yes_or_no_batch = []
            yes_or_no_torch_labels_cross_entropy = []
            sentence = []

            for i in input_ids:
                sentence.append(
                    self.tokenizer_large.decode(i, skip_special_tokens=True).replace('What aspect terms?', '').strip())

            for i in range(batch_size):
                all_aspects_in_a_sentence = []
                for j in pred_aspects[i]:
                    all_aspects_in_a_sentence.append(j)

                question_input = []
                count_list = self.generate_count_list(lst=all_aspects_in_a_sentence)

                for aspect, count in zip(all_aspects_in_a_sentence, count_list):
                    join_aspect = ' '.join(aspect)
                    front = self.question[:self.aspect_start_index] + " " + join_aspect + " " + self.question[
                                                                                                self.aspect_start_index:]

                    sentence_start_index = front.rfind('"')

                    modified_string = self.replace_nth_occurrence(s=sentence[i], target=join_aspect,
                                                                  replacement='<target> ' + join_aspect + ' </target>',
                                                                  n=count)

                    behind = front[:sentence_start_index] + modified_string + front[sentence_start_index:]
                    question_input.append(behind)

                try:
                    tokenized_question = self.processor(text=question_input, return_tensors="pt", truncation=True,
                                                        padding=True, max_length=self.question_max_length)
                except IndexError:
                    tokenized_question = self.processor(text="NULL", return_tensors="pt", truncation=True,
                                                        padding=True, max_length=self.question_max_length)

                vision_output = self.CLIPVisionEncoder(
                    pixel_values=pixel_values[i].unsqueeze(0).repeat(max(1, len(pred_aspects[i])), 1, 1, 1),
                    return_dict=self.config_clip.use_return_dict)

                text_output = self.CLIPTextEncoderWithProjection(input_ids=tokenized_question["input_ids"].cuda(),
                                                                 attention_mask=tokenized_question[
                                                                     "attention_mask"].cuda(),
                                                                 return_dict=self.config_clip.use_return_dict)

                align_q_feature, _ = self.attn(text_output["text_embeds"], vision_output["last_hidden_state"].detach())
                original_q_feature = self.q_feature_linear(text_output["text_embeds"])
                x = self.fusion(align_q_feature, original_q_feature)
                logits_yes_or_no = self.classifier_linear(x)
                yes_or_no_torch_labels = torch.tensor(yes_or_no_labels[i])
                yes_or_no_torch_labels_cross_entropy.append(yes_or_no_torch_labels)
                pred_labels_yes_or_no = torch.argmax(logits_yes_or_no, dim=-1)

                final_predicted_result = []
                final_predicted_result_pair = []

                for i_n_d_e_x, yes_or_no in enumerate(pred_labels_yes_or_no):
                    if yes_or_no == 1 and len(pred_aspects[i]) != 0:
                        final_predicted_result.append(pred_aspects[i][i_n_d_e_x])
                        final_predicted_result_pair.append(pred_pair_list_index_batch[i][i_n_d_e_x])

                final_predicted_result_batch.append(final_predicted_result)
                final_predicted_result_pair_batch.append(final_predicted_result_pair)
                logits_yes_or_no_batch.append(logits_yes_or_no)
                pred_labels_yes_or_no_batch.append(pred_labels_yes_or_no)

            logits_yes_or_no_batch_cat = torch.cat(logits_yes_or_no_batch, dim=0)
            yes_or_no_torch_labels_cross_entropy_cat = torch.cat(yes_or_no_torch_labels_cross_entropy, dim=0).cuda()
            clip_loss = self.cross_entropy_loss(logits_yes_or_no_batch_cat,
                                                yes_or_no_torch_labels_cross_entropy_cat)

            if torch.isnan(clip_loss) == torch.tensor(True):
                clip_loss = torch.tensor(0.0, requires_grad=True).cuda()

            output_dict = {"pred_pair_list_index_batch": pred_pair_list_index_batch,
                           "pred_labels_yes_or_no_batch": pred_labels_yes_or_no_batch, "clip_loss": clip_loss,
                           "logits_yes_or_no_batch_cat": logits_yes_or_no_batch_cat}

            self.to_tensor(output_dict)
            self.padding_and_truncation(output_tensor=output_dict)
            return output_dict

        def generate_labels(self, pred_aspects, gold_aspects):
            labels = []

            for pred_list, gold_list in zip(pred_aspects, gold_aspects):
                label_list = []
                for pred_item in pred_list:
                    if pred_item in gold_list:
                        label_list.append(1)
                    else:
                        label_list.append(0)
                labels.append(label_list)

            return labels

        def add_value_to_empty_sublists(self, lst, value):
            for sublst in lst:
                if not sublst:
                    sublst.append(value)
            return lst

        def to_tensor(self, output_dict):
            for i, item in enumerate(output_dict["pred_pair_list_index_batch"]):
                for j, inner_item in enumerate(item):
                    output_dict["pred_pair_list_index_batch"][i][j] = torch.tensor(inner_item)

        def padding_and_truncation(self, output_tensor):
            pred_labels_yes_or_no_batch = output_tensor["pred_labels_yes_or_no_batch"]

            big_tensor_three_format = self.get_big_tensor_three_format(tensors=pred_labels_yes_or_no_batch,
                                                                       target_length=self.aspect_term_num)

            output_tensor["pred_labels_yes_or_no_batch"] = big_tensor_three_format

        def get_big_tensor_three_format(self, tensors, target_length):
            padded_value = -200

            padded_tensors = []
            for tensor in tensors:
                padding_size = target_length - tensor.size(0)

                if padding_size > 0:
                    padded_tensor = F.pad(tensor, (0, padding_size), value=padded_value)
                else:
                    padded_tensor = tensor[:target_length]

                padded_tensors.append(padded_tensor)

            stacked_tensor = torch.stack(padded_tensors)
            return stacked_tensor

        def generate_count_list(self, lst):
            count_dict = {}
            result = []
            for item in lst:
                element = tuple(item)
                if element in count_dict:
                    count_dict[element] += 1
                else:
                    count_dict[element] = 1
                result.append(count_dict[element])
            return result

        def replace_nth_occurrence(self, s, target, replacement, n):
            index = -1
            for _ in range(n):
                index = s.find(target, index + 1)
                if index == -1:
                    return s
            return s[:index] + replacement + s[index + len(target):]

    def __init__(self):
        super(Twitter2015MATEDecisionAddSentence_max_length_Fixed_padding_target_DeBERTaModel, self).__init__()
        self.Twitter2015MATECLIPDropoutCrossEntropyLossQuestionRoBERTa_large_DropoutCRFModel_inner_class = self.Twitter2015MATECLIPDropoutCrossEntropyLossQuestionRoBERTa_large_DropoutCRFModel()
        self.Twitter2015MATEQuestionRoBERTa_large_DropoutCRFModel_inner_class = self.Twitter2015MATEQuestionRoBERTa_large_DropoutCRFModel()
        self.Twitter2015MATEDescription_pooler_output_DropoutCrossEntropyLossQuestionRoBERTa_large_DropoutCRFModel_inner_class = self.Twitter2015MATEDescription_pooler_output_DropoutCrossEntropyLossQuestionRoBERTa_large_DropoutCRFModel()

        self.aspect_term_num = 60
        self.aspect_term_max_length = 60
        self.pretrained_model = "pretrained_model"
        self.text_pretrained_model_large = 'deberta-v3-large'

        self.pretrained_model_path_large = os.path.join(self.pretrained_model, self.text_pretrained_model_large)

        self.tokenizer_large = AutoTokenizer.from_pretrained(self.pretrained_model_path_large,
                                                             add_prefix_space=True)

    def forward(self, MATEinput_ids, MATEattention_mask, MATElabels, MATEcross_labels, MATEpixel_values, MATEword_ids,
                MATEdescription_input_ids,
                **kwargs):
        output = self.Twitter2015MATEQuestionRoBERTa_large_DropoutCRFModel_inner_class(input_ids=MATEinput_ids,
                                                                                       attention_mask=MATEattention_mask,
                                                                                       labels=MATElabels,
                                                                                       cross_labels=MATEcross_labels)

        text_pred_labels = torch.argmax(output["logits"], dim=-1)

        pred_aspects, pred_pair_list_index_batch = self.get_aspects(input_ids=MATEinput_ids, word_ids=MATEword_ids,
                                                                    text_pred_labels=text_pred_labels)

        gold_aspects, gold_pair_list_index_batch = self.get_aspects(input_ids=MATEinput_ids, word_ids=MATEword_ids,
                                                                    text_pred_labels=MATEcross_labels)

        clip_output = self.Twitter2015MATECLIPDropoutCrossEntropyLossQuestionRoBERTa_large_DropoutCRFModel_inner_class(
            input_ids=MATEinput_ids, pixel_values=MATEpixel_values, output=output,
            pred_aspects=copy.deepcopy(pred_aspects),
            pred_pair_list_index_batch=copy.deepcopy(pred_pair_list_index_batch),
            gold_aspects=copy.deepcopy(gold_aspects),
            gold_pair_list_index_batch=copy.deepcopy(gold_pair_list_index_batch))

        description_output = self.Twitter2015MATEDescription_pooler_output_DropoutCrossEntropyLossQuestionRoBERTa_large_DropoutCRFModel_inner_class(
            input_ids=MATEinput_ids, description_input_ids=MATEdescription_input_ids, output=output,
            pred_aspects=copy.deepcopy(pred_aspects),
            pred_pair_list_index_batch=copy.deepcopy(pred_pair_list_index_batch),
            gold_aspects=copy.deepcopy(gold_aspects),
            gold_pair_list_index_batch=copy.deepcopy(gold_pair_list_index_batch))

        clip_logits = clip_output['logits_yes_or_no_batch_cat']
        description_logits = description_output['logits_yes_or_no_batch_cat']
        decision_logits = F.softmax(clip_logits, dim=1) + F.softmax(description_logits, dim=1)

        clip_pred_labels_yes_or_no_batch = clip_output['pred_labels_yes_or_no_batch']

        clip_restored_list = [tensor[tensor != -200].tolist() for tensor in clip_pred_labels_yes_or_no_batch]

        self.replace_all_to_neg_one(lst=clip_restored_list)
        decision_pred_labels_yes_or_no = torch.argmax(decision_logits, dim=-1)
        self.replace_nested_list_values(nested_list=clip_restored_list, values_tensor=decision_pred_labels_yes_or_no)
        clip_pred_pair_list_index_batch = clip_output['pred_pair_list_index_batch']
        filtered_result = self.filter_lists_v3(input_list=clip_pred_pair_list_index_batch,
                                               filter_mask=clip_restored_list)

        big_tensor_two_format = self.get_big_tensor_two_format(tensor_groups=filtered_result,
                                                               max_length=self.aspect_term_max_length,
                                                               max_num=self.aspect_term_num)

        loss = output["loss"] + clip_output['clip_loss'] + description_output['description_loss']

        merge_output = {"decision_final_predicted_result_pair_batch": big_tensor_two_format, "loss": loss}

        return merge_output

    def replace_all_to_neg_one(self, lst):
        for i in range(len(lst)):
            if isinstance(lst[i], list):
                self.replace_all_to_neg_one(lst[i])
            else:
                lst[i] = -1

    def replace_nested_list_values(self, nested_list, values_tensor):
        value_index = 0
        for sub_list in nested_list:
            for i in range(len(sub_list)):
                if value_index < len(values_tensor):
                    sub_list[i] = values_tensor[value_index].item()
                    value_index += 1

    def get_big_tensor_three_format(self, tensors, target_length):
        padded_value = -200

        padded_tensors = []
        for tensor in tensors:
            padding_size = target_length - tensor.size(0)

            if padding_size > 0:
                padded_tensor = F.pad(tensor, (0, padding_size), value=padded_value)
            else:
                padded_tensor = tensor[:target_length]

            padded_tensors.append(padded_tensor)

        stacked_tensor = torch.stack(padded_tensors)
        return stacked_tensor

    def filter_lists_v3(self, input_list, filter_mask):
        def filter_recursive(sublist, mask):
            if isinstance(mask, list):
                filtered = [filter_recursive(sublist[i], mask[i]) for i in range(len(mask)) if i < len(sublist)]
                return [item for item in filtered if item is not None]
            else:
                return sublist if mask == 1 else None

        return filter_recursive(input_list, filter_mask)

    def get_big_tensor_two_format(self, tensor_groups, max_length, max_num):
        processed_group_tensors = []

        for group in tensor_groups:
            new_tensor = torch.full((max_num, max_length), -200)

            for index, tensor in enumerate(group):
                if index < max_num:
                    if tensor.size(0) < max_length:
                        padding_size = max_length - tensor.size(0)
                        padded_tensor = F.pad(tensor, (0, padding_size), value=-200)
                    elif tensor.size(0) > max_length:
                        padded_tensor = tensor[:max_length]
                    else:
                        padded_tensor = tensor

                    new_tensor[index] = padded_tensor

            processed_group_tensors.append(new_tensor)

        big_tensor_group = torch.stack(processed_group_tensors)
        return big_tensor_group

    def get_aspects(self, input_ids, word_ids, text_pred_labels):
        pred_pair_list = self.get_predicted_pair(p_pred_labels=text_pred_labels, word_ids=word_ids)
        pred_pair_list_index_batch = []
        pred_pair_list_index_single = []

        for i in pred_pair_list:
            for j in i:
                pred_pair_list_index_single.append(list(range(j[0], j[1] + 1)))
            pred_pair_list_index_batch.append(pred_pair_list_index_single)
            pred_pair_list_index_single = []

        original_sentence = []

        for i in input_ids:
            original_sentence.append(self.tokenizer_large.tokenize(self.tokenizer_large.decode(i)))

        positions_single = []
        positions_batch = []

        for index, spans in enumerate(pred_pair_list_index_batch):
            for single_span in spans:
                positions_single.append(
                    [i for i, value in enumerate(word_ids[index]) if value in single_span])
            positions_batch.append(positions_single)
            positions_single = []

        aspect_term = []
        selected_data = []

        for positions, sentence in zip(positions_batch, original_sentence):
            for position in positions:
                selected_data.append([sentence[i] for i in position])
            aspect_term.append(selected_data)
            selected_data = []

        aspects = self.merge_tokens_three(nested_list=copy.deepcopy(aspect_term))

        return aspects, pred_pair_list_index_batch

    def get_predicted_pair(self, p_pred_labels, word_ids):
        pred_pair_list = []
        for i, pred_label in enumerate(p_pred_labels):
            word_ids_in_word_ids = word_ids[i]
            flag = False
            pred_pair = set()
            sentiment = 0
            start_pos = 0
            end_pos = 0
            for j, pp in enumerate(pred_label):
                if word_ids_in_word_ids[j] == -100:
                    if flag:
                        pred_pair.add((start_pos, end_pos,
                                       sentiment))
                        flag = False
                    continue
                if word_ids_in_word_ids[j] != word_ids_in_word_ids[
                    j - 1]:
                    if pp > 1:
                        if flag:
                            pred_pair.add((start_pos, end_pos,
                                           sentiment))
                        start_pos = word_ids_in_word_ids[j]
                        end_pos = word_ids_in_word_ids[j]
                        sentiment = pp - 2
                        flag = True
                    elif pp == 1:
                        if flag:
                            end_pos = word_ids_in_word_ids[
                                j]
                    else:
                        if flag:
                            pred_pair.add((start_pos, end_pos,
                                           sentiment))
                        flag = False
            pred_pair_list.append(pred_pair.copy())

        return pred_pair_list

    def merge_tokens_three(self, nested_list):
        for sub_list in nested_list:
            for token_list in sub_list:
                i = 1
                while i < len(token_list):
                    if not token_list[i].startswith("▁"):
                        token_list[i - 1] += token_list[i]
                        token_list.pop(i)
                    else:
                        i += 1

        result_list = [[[s[1:] if s.startswith('▁') else s for s in sublist] for sublist in subsublist] for
                       subsublist
                       in nested_list]
        return result_list


class Twitter2015MASCDeBERTa_large_target_Model(nn.Module):
    def __init__(self):
        super(Twitter2015MASCDeBERTa_large_target_Model, self).__init__()
        self.pretrained_model = "pretrained_model"
        self.padding_strategy = "max_length"
        self.question = ["What", "is", "the", "sentiment", "polarity", "of", "the", "<target>", "</target>", "?"]
        self.aspect_term_max_length = 60
        self.aspect_term_num = 60
        self.text_pretrained_model_large = 'deberta-v3-large'
        self.question_max_length = 80
        self.special_tokens_dict = {'additional_special_tokens': ['<target>', '</target>']}

        self.pretrained_model_path_large = os.path.join(self.pretrained_model, self.text_pretrained_model_large)
        self.config_roberta = AutoConfig.from_pretrained(self.pretrained_model_path_large)
        self.aspect_start_index = self.question.index("<target>") + 1

        self.tokenizer_large = AutoTokenizer.from_pretrained(self.pretrained_model_path_large, add_prefix_space=True)
        self.tokenizer_large.add_special_tokens(self.special_tokens_dict)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(self.config_roberta.hidden_size, 3)
        self.dropout = nn.Dropout(self.config_roberta.hidden_dropout_prob)
        self.RoBERTa_large = AutoModel.from_pretrained(self.pretrained_model_path_large)
        self.RoBERTa_large.resize_token_embeddings(len(self.tokenizer_large))

    def forward(self, input_ids, cross_labels, word_ids, **kwargs):
        gold_aspects, gold_pair_list_index_batch, gold_sentiments = self.get_aspects(input_ids=input_ids,
                                                                                     word_ids=word_ids,
                                                                                     text_pred_labels=cross_labels)

        sentence = []

        for i in input_ids:
            sentence.append(self.tokenizer_large.tokenize(self.tokenizer_large.decode(i, skip_special_tokens=True)))

        sentence_merge = self.merge_tokens_two(nested_list=copy.deepcopy(sentence))
        batch_size = input_ids.shape[0]
        logits_batch = []
        torch_labels_cross_entropy = []

        for i in range(batch_size):
            all_aspects_in_a_sentence = []
            for j in gold_aspects[i]:
                all_aspects_in_a_sentence.append(j)

            question_input = []
            sentence_input = []
            count_list = self.generate_count_list(lst=all_aspects_in_a_sentence)

            for aspect, count in zip(all_aspects_in_a_sentence, count_list):
                question_input.append(
                    self.question[:self.aspect_start_index] + aspect + self.question[self.aspect_start_index:])
                sentence_input.append(
                    self.insert_tags_around_sequence(lst=sentence_merge[i].copy(), sequence=aspect, occurrence=count))

            tokenized_sentence = self.tokenizer_large(question_input, sentence_input,
                                                      return_tensors="pt", truncation=True,
                                                      padding=self.padding_strategy,
                                                      max_length=self.question_max_length, is_split_into_words=True)

            sentence_output = self.RoBERTa_large(input_ids=tokenized_sentence["input_ids"].cuda(),
                                                 attention_mask=tokenized_sentence["attention_mask"].cuda(),
                                                 return_dict=self.config_roberta.use_return_dict)

            logits = self.classifier(sentence_output["last_hidden_state"][:, 8, :])
            logits_dropout = self.dropout(logits)
            logits_batch.append(logits_dropout)
            torch_labels = torch.tensor(gold_sentiments[i])
            torch_labels_cross_entropy.append(torch_labels)

        logits_batch_cat = torch.cat(logits_batch, dim=0)
        torch_labels_cross_entropy_cat = torch.cat(torch_labels_cross_entropy, dim=0).cuda()
        loss = self.cross_entropy_loss(logits_batch_cat, torch_labels_cross_entropy_cat)

        output_dict = {"loss": loss, "logits": logits_batch_cat,
                       "gold_pair_list_index_batch": gold_pair_list_index_batch,
                       "torch_labels_cross_entropy_cat": torch_labels_cross_entropy_cat}

        self.to_tensor(output_dict)
        self.padding_and_truncation(output_tensor=output_dict)

        return output_dict

    def get_predicted_pair(self, p_pred_labels, word_ids):
        pred_pair_list = []
        for i, pred_label in enumerate(p_pred_labels):
            word_ids_in_word_ids = word_ids[i]
            flag = False
            pred_pair = set()
            sentiment = 0
            start_pos = 0
            end_pos = 0
            for j, pp in enumerate(pred_label):
                if word_ids_in_word_ids[j] == -100:
                    if flag:
                        pred_pair.add((start_pos, end_pos,
                                       sentiment))
                        flag = False
                    continue
                if word_ids_in_word_ids[j] != word_ids_in_word_ids[
                    j - 1]:
                    if pp > 1:
                        if flag:
                            pred_pair.add((start_pos, end_pos,
                                           sentiment))
                        start_pos = word_ids_in_word_ids[j]
                        end_pos = word_ids_in_word_ids[j]
                        sentiment = pp - 2
                        flag = True
                    elif pp == 1:
                        if flag:
                            end_pos = word_ids_in_word_ids[
                                j]
                    else:
                        if flag:
                            pred_pair.add((start_pos, end_pos,
                                           sentiment))
                        flag = False
            pred_pair_list.append(pred_pair.copy())

        return pred_pair_list

    def merge_tokens_three(self, nested_list):
        for sub_list in nested_list:
            for token_list in sub_list:
                i = 1
                while i < len(token_list):
                    if not token_list[i].startswith("▁"):
                        token_list[i - 1] += token_list[i]
                        token_list.pop(i)
                    else:
                        i += 1

        result_list = [[[s[1:] if s.startswith('▁') else s for s in sublist] for sublist in subsublist] for subsublist
                       in nested_list]
        return result_list

    def get_aspects(self, input_ids, word_ids, text_pred_labels):
        pred_pair_list = self.get_predicted_pair(p_pred_labels=text_pred_labels, word_ids=word_ids)
        pred_pair_list_index_batch = []
        pred_pair_list_index_single = []

        for i in pred_pair_list:
            for j in i:
                pred_pair_list_index_single.append(list(range(j[0], j[1] + 1)) + [j[2]])
            pred_pair_list_index_batch.append(pred_pair_list_index_single)
            pred_pair_list_index_single = []

        pred_pair_list_index_batch, gold_sentiments = self.separate_tensors(data=pred_pair_list_index_batch)

        original_sentence = []

        for i in input_ids:
            original_sentence.append(self.tokenizer_large.tokenize(self.tokenizer_large.decode(i)))

        positions_single = []
        positions_batch = []

        for index, spans in enumerate(pred_pair_list_index_batch):
            for single_span in spans:
                positions_single.append(
                    [i for i, value in enumerate(word_ids[index]) if value in single_span])
            positions_batch.append(positions_single)
            positions_single = []

        aspect_term = []
        selected_data = []

        for positions, sentence in zip(positions_batch, original_sentence):
            for position in positions:
                selected_data.append([sentence[i] for i in position])
            aspect_term.append(selected_data)
            selected_data = []

        aspects = self.merge_tokens_three(nested_list=copy.deepcopy(aspect_term))

        return aspects, pred_pair_list_index_batch, gold_sentiments

    def to_tensor(self, output_dict):
        for i, item in enumerate(output_dict["gold_pair_list_index_batch"]):
            for j, inner_item in enumerate(item):
                output_dict["gold_pair_list_index_batch"][i][j] = torch.tensor(inner_item)

    def padding_and_truncation(self, output_tensor):
        gold_pair_list_index_batch = output_tensor["gold_pair_list_index_batch"]

        big_tensor_two_format = self.get_big_tensor_two_format(tensor_groups=gold_pair_list_index_batch,
                                                               max_length=self.aspect_term_max_length,
                                                               max_num=self.aspect_term_num)

        output_tensor["gold_pair_list_index_batch"] = big_tensor_two_format

    def merge_tokens_two(self, nested_list):
        for sub_list in nested_list:
            i = 1
            while i < len(sub_list):
                if not sub_list[i].startswith("▁"):
                    sub_list[i - 1] += sub_list[i]
                    sub_list.pop(i)
                else:
                    i += 1

        result_list = [[s[1:] if s.startswith('▁') else s for s in sublist] for sublist in nested_list]

        return result_list

    def get_big_tensor_two_format(self, tensor_groups, max_length, max_num):
        processed_group_tensors = []

        for group in tensor_groups:
            new_tensor = torch.full((max_num, max_length), -200)

            for index, tensor in enumerate(group):
                if index < max_num:
                    if tensor.size(0) < max_length:
                        padding_size = max_length - tensor.size(0)
                        padded_tensor = F.pad(tensor, (0, padding_size), value=-200)
                    elif tensor.size(0) > max_length:
                        padded_tensor = tensor[:max_length]
                    else:
                        padded_tensor = tensor

                    new_tensor[index] = padded_tensor

            processed_group_tensors.append(new_tensor)

        big_tensor_group = torch.stack(processed_group_tensors)
        return big_tensor_group

    def separate_tensors(self, data):
        values_without_tensors = []
        tensors = []

        for sublist in data:
            current_sublist = []
            current_tensors = []
            for item in sublist:
                current_sublist.append(item[:-1])
                current_tensors.append(item[-1])
            values_without_tensors.append(current_sublist)
            tensors.append(current_tensors)

        return values_without_tensors, tensors

    def generate_count_list(self, lst):
        count_dict = {}
        result = []
        for item in lst:
            element = tuple(item)
            if element in count_dict:
                count_dict[element] += 1
            else:
                count_dict[element] = 1
            result.append(count_dict[element])
        return result

    def insert_tags_around_sequence(self, lst, sequence, occurrence):
        length = len(sequence)
        count = 0

        i = 0
        while i <= len(lst) - length:
            if lst[i:i + length] == sequence:
                count += 1
                if count == occurrence:
                    lst.insert(i, '<target>')
                    lst.insert(i + length + 1, '</target>')
                    break
            i += 1

        return lst


class Twitter2015MASCDescriptionDeBERTa_large_target_Model(nn.Module):
    def __init__(self):
        super(Twitter2015MASCDescriptionDeBERTa_large_target_Model, self).__init__()
        self.pretrained_model = "pretrained_model"
        self.padding_strategy = "max_length"
        self.question = "What is the sentiment polarity of the <target></target> in the sentence \"\"?"
        self.aspect_term_max_length = 60
        self.aspect_term_num = 60
        self.text_pretrained_model_large = 'deberta-v3-large'
        self.question_max_length = 512
        self.special_tokens_dict = {'additional_special_tokens': ['<target>', '</target>']}

        self.pretrained_model_path_large = os.path.join(self.pretrained_model, self.text_pretrained_model_large)
        self.config_roberta = AutoConfig.from_pretrained(self.pretrained_model_path_large)
        self.aspect_start_index = self.question.index(">") + 1

        self.tokenizer_large = AutoTokenizer.from_pretrained(self.pretrained_model_path_large, add_prefix_space=True)
        self.tokenizer_large.add_special_tokens(self.special_tokens_dict)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(self.config_roberta.hidden_size, 3)
        self.dropout = nn.Dropout(self.config_roberta.hidden_dropout_prob)
        self.RoBERTa_large = AutoModel.from_pretrained(self.pretrained_model_path_large)
        self.RoBERTa_large.resize_token_embeddings(len(self.tokenizer_large))

    def forward(self, input_ids, cross_labels, word_ids, description_input_ids, **kwargs):
        gold_aspects, gold_pair_list_index_batch, gold_sentiments = self.get_aspects(input_ids=input_ids,
                                                                                     word_ids=word_ids,
                                                                                     text_pred_labels=cross_labels)

        sentence = []
        description_sentence = []

        for i in description_input_ids:
            description_sentence.append(self.tokenizer_large.decode(i, skip_special_tokens=True))

        for i in input_ids:
            sentence.append(
                self.tokenizer_large.decode(i, skip_special_tokens=True).strip())

        batch_size = input_ids.shape[0]
        logits_batch = []
        torch_labels_cross_entropy = []

        for i in range(batch_size):
            all_aspects_in_a_sentence = []
            for j in gold_aspects[i]:
                all_aspects_in_a_sentence.append(j)

            question_input = []
            count_list = self.generate_count_list(lst=all_aspects_in_a_sentence)

            for aspect, count in zip(all_aspects_in_a_sentence, count_list):
                join_aspect = ' '.join(aspect)
                front = self.question[:self.aspect_start_index] + " " + join_aspect + " " + self.question[
                                                                                            self.aspect_start_index:]

                sentence_start_index = front.rfind('"')

                modified_string = self.replace_nth_occurrence(s=sentence[i], target=join_aspect,
                                                              replacement='<target> ' + join_aspect + ' </target>',
                                                              n=count)

                behind = front[:sentence_start_index] + modified_string + front[sentence_start_index:]
                question_input.append(behind)

            tokenized_sentence = self.tokenizer_large(question_input, [description_sentence[i]] * len(gold_aspects[i]),
                                                      return_tensors="pt", truncation=True,
                                                      padding=self.padding_strategy,
                                                      max_length=self.question_max_length)

            sentence_output = self.RoBERTa_large(input_ids=tokenized_sentence["input_ids"].cuda(),
                                                 attention_mask=tokenized_sentence["attention_mask"].cuda(),
                                                 return_dict=self.config_roberta.use_return_dict)

            logits = self.classifier(sentence_output["last_hidden_state"][:, 8, :])
            logits_dropout = self.dropout(logits)
            logits_batch.append(logits_dropout)
            torch_labels = torch.tensor(gold_sentiments[i])
            torch_labels_cross_entropy.append(torch_labels)

        logits_batch_cat = torch.cat(logits_batch, dim=0)
        torch_labels_cross_entropy_cat = torch.cat(torch_labels_cross_entropy, dim=0).cuda()
        loss = self.cross_entropy_loss(logits_batch_cat, torch_labels_cross_entropy_cat)

        output_dict = {"loss": loss, "logits": logits_batch_cat,
                       "gold_pair_list_index_batch": gold_pair_list_index_batch,
                       "torch_labels_cross_entropy_cat": torch_labels_cross_entropy_cat}

        self.to_tensor(output_dict)
        self.padding_and_truncation(output_tensor=output_dict)

        return output_dict

    def get_predicted_pair(self, p_pred_labels, word_ids):
        pred_pair_list = []
        for i, pred_label in enumerate(p_pred_labels):
            word_ids_in_word_ids = word_ids[i]
            flag = False
            pred_pair = set()
            sentiment = 0
            start_pos = 0
            end_pos = 0
            for j, pp in enumerate(pred_label):
                if word_ids_in_word_ids[j] == -100:
                    if flag:
                        pred_pair.add((start_pos, end_pos,
                                       sentiment))
                        flag = False
                    continue
                if word_ids_in_word_ids[j] != word_ids_in_word_ids[
                    j - 1]:
                    if pp > 1:
                        if flag:
                            pred_pair.add((start_pos, end_pos,
                                           sentiment))
                        start_pos = word_ids_in_word_ids[j]
                        end_pos = word_ids_in_word_ids[j]
                        sentiment = pp - 2
                        flag = True
                    elif pp == 1:
                        if flag:
                            end_pos = word_ids_in_word_ids[
                                j]
                    else:
                        if flag:
                            pred_pair.add((start_pos, end_pos,
                                           sentiment))
                        flag = False
            pred_pair_list.append(pred_pair.copy())

        return pred_pair_list

    def merge_tokens_three(self, nested_list):
        for sub_list in nested_list:
            for token_list in sub_list:
                i = 1
                while i < len(token_list):
                    if not token_list[i].startswith("▁"):
                        token_list[i - 1] += token_list[i]
                        token_list.pop(i)
                    else:
                        i += 1

        result_list = [[[s[1:] if s.startswith('▁') else s for s in sublist] for sublist in subsublist] for subsublist
                       in nested_list]
        return result_list

    def get_aspects(self, input_ids, word_ids, text_pred_labels):
        pred_pair_list = self.get_predicted_pair(p_pred_labels=text_pred_labels, word_ids=word_ids)
        pred_pair_list_index_batch = []
        pred_pair_list_index_single = []

        for i in pred_pair_list:
            for j in i:
                pred_pair_list_index_single.append(list(range(j[0], j[1] + 1)) + [j[2]])
            pred_pair_list_index_batch.append(pred_pair_list_index_single)
            pred_pair_list_index_single = []

        pred_pair_list_index_batch, gold_sentiments = self.separate_tensors(data=pred_pair_list_index_batch)

        original_sentence = []

        for i in input_ids:
            original_sentence.append(self.tokenizer_large.tokenize(self.tokenizer_large.decode(i)))

        positions_single = []
        positions_batch = []

        for index, spans in enumerate(pred_pair_list_index_batch):
            for single_span in spans:
                positions_single.append(
                    [i for i, value in enumerate(word_ids[index]) if value in single_span])
            positions_batch.append(positions_single)
            positions_single = []

        aspect_term = []
        selected_data = []

        for positions, sentence in zip(positions_batch, original_sentence):
            for position in positions:
                selected_data.append([sentence[i] for i in position])
            aspect_term.append(selected_data)
            selected_data = []

        aspects = self.merge_tokens_three(nested_list=copy.deepcopy(aspect_term))

        return aspects, pred_pair_list_index_batch, gold_sentiments

    def to_tensor(self, output_dict):
        for i, item in enumerate(output_dict["gold_pair_list_index_batch"]):
            for j, inner_item in enumerate(item):
                output_dict["gold_pair_list_index_batch"][i][j] = torch.tensor(inner_item)

    def padding_and_truncation(self, output_tensor):
        gold_pair_list_index_batch = output_tensor["gold_pair_list_index_batch"]

        big_tensor_two_format = self.get_big_tensor_two_format(tensor_groups=gold_pair_list_index_batch,
                                                               max_length=self.aspect_term_max_length,
                                                               max_num=self.aspect_term_num)

        output_tensor["gold_pair_list_index_batch"] = big_tensor_two_format

    def get_big_tensor_two_format(self, tensor_groups, max_length, max_num):
        processed_group_tensors = []

        for group in tensor_groups:
            new_tensor = torch.full((max_num, max_length), -200)

            for index, tensor in enumerate(group):
                if index < max_num:
                    if tensor.size(0) < max_length:
                        padding_size = max_length - tensor.size(0)
                        padded_tensor = F.pad(tensor, (0, padding_size), value=-200)
                    elif tensor.size(0) > max_length:
                        padded_tensor = tensor[:max_length]
                    else:
                        padded_tensor = tensor

                    new_tensor[index] = padded_tensor

            processed_group_tensors.append(new_tensor)

        big_tensor_group = torch.stack(processed_group_tensors)
        return big_tensor_group

    def separate_tensors(self, data):
        values_without_tensors = []
        tensors = []

        for sublist in data:
            current_sublist = []
            current_tensors = []
            for item in sublist:
                current_sublist.append(item[:-1])
                current_tensors.append(item[-1])
            values_without_tensors.append(current_sublist)
            tensors.append(current_tensors)

        return values_without_tensors, tensors

    def generate_count_list(self, lst):
        count_dict = {}
        result = []
        for item in lst:
            element = tuple(item)
            if element in count_dict:
                count_dict[element] += 1
            else:
                count_dict[element] = 1
            result.append(count_dict[element])
        return result

    def replace_nth_occurrence(self, s, target, replacement, n):
        index = -1
        for _ in range(n):
            index = s.find(target, index + 1)
            if index == -1:
                return s
        return s[:index] + replacement + s[index + len(target):]


class Twitter2015MASCCLIP_large_336_target_DeBERTaModel(nn.Module):
    class MFBFusion_outer(nn.Module):
        def __init__(self, input_dim1, input_dim2, hidden_dim, R):
            super().__init__()
            self.input_dim1 = input_dim1
            self.input_dim2 = input_dim2
            self.hidden_dim = hidden_dim
            self.R = R
            self.linear1 = nn.Linear(input_dim1, hidden_dim * R)
            self.linear2 = nn.Linear(input_dim2, hidden_dim * R)

        def forward(self, inputs1, inputs2):
            num_region = 1
            if inputs1.dim() == 3:
                num_region = inputs1.size(1)
            h1 = self.linear1(inputs1)
            h2 = self.linear2(inputs2)
            z = h1 * h2
            z = z.view(z.size(0), num_region, self.hidden_dim, self.R)
            z = z.sum(3).squeeze(1)
            return z

    class MultiHeadATTN(nn.Module):
        class MFBFusion_inner(nn.Module):
            def __init__(self, input_dim1, input_dim2, hidden_dim, R):
                super().__init__()
                self.input_dim1 = input_dim1
                self.input_dim2 = input_dim2
                self.hidden_dim = hidden_dim
                self.R = R
                self.linear1 = nn.Linear(input_dim1, hidden_dim * R)
                self.linear2 = nn.Linear(input_dim2, hidden_dim * R)

            def forward(self, inputs1, inputs2):
                num_region = 1
                if inputs1.dim() == 3:
                    num_region = inputs1.size(1)
                h1 = self.linear1(inputs1)
                h2 = self.linear2(inputs2)
                z = h1 * h2
                z = z.view(z.size(0), num_region, self.hidden_dim, self.R)
                z = z.sum(3).squeeze(1)
                return z

        def __init__(self, query_dim, kv_dim, mfb_input_dim, mfb_hidden_dim, num_head, att_dim):
            super().__init__()
            assert att_dim % num_head == 0
            self.num_head = num_head
            self.att_dim = att_dim
            self.R = 1

            self.attn_w_1_q = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(query_dim, mfb_input_dim),
                nn.ReLU()
            )

            self.attn_w_1_k = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(kv_dim, mfb_input_dim),
                nn.ReLU()
            )

            self.attn_score_fusion = self.MFBFusion_inner(mfb_input_dim, mfb_input_dim, mfb_hidden_dim, self.R)

            self.attn_score_mapping = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(mfb_hidden_dim, num_head)
            )

            self.softmax = nn.Softmax(dim=1)

            self.align_q = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(kv_dim, round(att_dim / num_head)),
                        nn.Tanh()
                    ) for _ in range(num_head)
                ]
            )

        def forward(self, query, key_value):
            num_region = key_value.shape[1]
            q = self.attn_w_1_q(query).unsqueeze(1).repeat(1, num_region, 1)
            k = self.attn_w_1_k(key_value)
            alphas = self.attn_score_fusion(q, k)
            alphas = self.attn_score_mapping(alphas)
            alphas = self.softmax(alphas)
            output = torch.bmm(alphas.transpose(1, 2), key_value)
            list_v = [e.squeeze(dim=1) for e in torch.split(output, 1, dim=1)]
            alpha = torch.split(alphas, 1, dim=2)
            align_feat = [self.align_q[head_id](x_v) for head_id, x_v in enumerate(list_v)]
            align_feat = torch.cat(align_feat, 1)
            return align_feat, alpha

    def __init__(self):
        super().__init__()
        self.pretrained_model = "pretrained_model"
        self.question_max_length = 77
        self.padding_strategy = "max_length"
        self.question = "What is the sentiment polarity of the <target></target> in the sentence \"\"?"
        self.aspect_term_max_length = 60
        self.aspect_term_num = 60
        self.text_pretrained_model_large = 'deberta-v3-large'
        self.CLIP = 'clip-vit-large-patch14-336'
        self.R = 8

        self.pretrained_model_path_large = os.path.join(self.pretrained_model, self.text_pretrained_model_large)
        self.aspect_start_index = self.question.index(">") + 1
        self.pretrained_model_path_clip = os.path.join(self.pretrained_model, self.CLIP)
        self.config_clip = AutoConfig.from_pretrained(self.pretrained_model_path_clip)
        self.attn_mfb_input_dim = round(
            self.config_clip.projection_dim / self.config_clip.text_config.num_attention_heads)
        self.attn_mfb_hidden_dim = self.attn_mfb_input_dim
        self.attn_output_dim = self.config_clip.text_config.num_attention_heads * self.attn_mfb_input_dim
        self.fusion_q_feature_dim = self.attn_mfb_input_dim
        self.fusion_mfb_hidden_dim = self.attn_mfb_hidden_dim

        self.tokenizer_large = AutoTokenizer.from_pretrained(self.pretrained_model_path_large, add_prefix_space=True)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.processor = AutoProcessor.from_pretrained(self.pretrained_model_path_clip)
        self.CLIPVisionEncoder = CLIPVisionModel.from_pretrained(self.pretrained_model_path_clip)
        self.CLIPTextEncoderWithProjection = CLIPTextModelWithProjection.from_pretrained(
            self.pretrained_model_path_clip)
        self.classifier_linear = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.fusion_mfb_hidden_dim, 3))
        self.fusion = self.MFBFusion_outer(input_dim1=self.attn_output_dim,
                                           input_dim2=self.fusion_q_feature_dim,
                                           hidden_dim=self.fusion_mfb_hidden_dim, R=self.R)
        self.attn = self.MultiHeadATTN(query_dim=self.config_clip.projection_dim,
                                       kv_dim=self.config_clip.vision_config.hidden_size,
                                       mfb_input_dim=self.attn_mfb_input_dim, mfb_hidden_dim=self.attn_mfb_hidden_dim,
                                       num_head=self.config_clip.text_config.num_attention_heads,
                                       att_dim=self.attn_output_dim)
        self.q_feature_linear = nn.Sequential(nn.Dropout(0.5),
                                              nn.Linear(self.config_clip.projection_dim,
                                                        self.fusion_q_feature_dim), nn.ReLU())

    def forward(self, input_ids, cross_labels, word_ids, pixel_values, **kwargs):
        gold_aspects, gold_pair_list_index_batch, gold_sentiments = self.get_aspects(input_ids=input_ids,
                                                                                     word_ids=word_ids,
                                                                                     text_pred_labels=cross_labels)

        sentence = []

        for i in input_ids:
            sentence.append(
                self.tokenizer_large.decode(i, skip_special_tokens=True).strip())

        batch_size = input_ids.shape[0]
        logits_batch = []
        torch_labels_cross_entropy = []

        for i in range(batch_size):
            all_aspects_in_a_sentence = []
            for j in gold_aspects[i]:
                all_aspects_in_a_sentence.append(j)

            question_input = []
            count_list = self.generate_count_list(lst=all_aspects_in_a_sentence)

            for aspect, count in zip(all_aspects_in_a_sentence, count_list):
                join_aspect = ' '.join(aspect)
                front = self.question[:self.aspect_start_index] + " " + join_aspect + " " + self.question[
                                                                                            self.aspect_start_index:]

                sentence_start_index = front.rfind('"')

                modified_string = self.replace_nth_occurrence(s=sentence[i], target=join_aspect,
                                                              replacement='<target> ' + join_aspect + ' </target>',
                                                              n=count)

                behind = front[:sentence_start_index] + modified_string + front[sentence_start_index:]
                question_input.append(behind)

            tokenized_question = self.processor(text=question_input, return_tensors="pt", truncation=True,
                                                padding=self.padding_strategy, max_length=self.question_max_length)

            vision_output = self.CLIPVisionEncoder(
                pixel_values=pixel_values[i].unsqueeze(0).repeat(len(gold_aspects[i]), 1, 1, 1),
                return_dict=self.config_clip.use_return_dict)

            text_output = self.CLIPTextEncoderWithProjection(input_ids=tokenized_question["input_ids"].cuda(),
                                                             attention_mask=tokenized_question["attention_mask"].cuda(),
                                                             return_dict=self.config_clip.use_return_dict)

            align_q_feature, _ = self.attn(text_output["text_embeds"], vision_output["last_hidden_state"].detach())
            original_q_feature = self.q_feature_linear(text_output["text_embeds"])
            x = self.fusion(align_q_feature, original_q_feature)
            logits = self.classifier_linear(x)
            logits_batch.append(logits)
            torch_labels = torch.tensor(gold_sentiments[i])
            torch_labels_cross_entropy.append(torch_labels)

        logits_batch_cat = torch.cat(logits_batch, dim=0)
        torch_labels_cross_entropy_cat = torch.cat(torch_labels_cross_entropy, dim=0).cuda()
        loss = self.cross_entropy_loss(logits_batch_cat, torch_labels_cross_entropy_cat)

        output_dict = {"loss": loss, "logits": logits_batch_cat,
                       "gold_pair_list_index_batch": gold_pair_list_index_batch,
                       "torch_labels_cross_entropy_cat": torch_labels_cross_entropy_cat}

        self.to_tensor(output_dict)
        self.padding_and_truncation(output_tensor=output_dict)

        return output_dict

    def get_predicted_pair(self, p_pred_labels, word_ids):
        pred_pair_list = []
        for i, pred_label in enumerate(p_pred_labels):
            word_ids_in_word_ids = word_ids[i]
            flag = False
            pred_pair = set()
            sentiment = 0
            start_pos = 0
            end_pos = 0
            for j, pp in enumerate(pred_label):
                if word_ids_in_word_ids[j] == -100:
                    if flag:
                        pred_pair.add((start_pos, end_pos,
                                       sentiment))
                        flag = False
                    continue
                if word_ids_in_word_ids[j] != word_ids_in_word_ids[
                    j - 1]:
                    if pp > 1:
                        if flag:
                            pred_pair.add((start_pos, end_pos,
                                           sentiment))
                        start_pos = word_ids_in_word_ids[j]
                        end_pos = word_ids_in_word_ids[j]
                        sentiment = pp - 2
                        flag = True
                    elif pp == 1:
                        if flag:
                            end_pos = word_ids_in_word_ids[
                                j]
                    else:
                        if flag:
                            pred_pair.add((start_pos, end_pos,
                                           sentiment))
                        flag = False
            pred_pair_list.append(pred_pair.copy())

        return pred_pair_list

    def merge_tokens_three(self, nested_list):
        for sub_list in nested_list:
            for token_list in sub_list:
                i = 1
                while i < len(token_list):
                    if not token_list[i].startswith("▁"):
                        token_list[i - 1] += token_list[i]
                        token_list.pop(i)
                    else:
                        i += 1

        result_list = [[[s[1:] if s.startswith('▁') else s for s in sublist] for sublist in subsublist] for subsublist
                       in nested_list]
        return result_list

    def get_aspects(self, input_ids, word_ids, text_pred_labels):
        pred_pair_list = self.get_predicted_pair(p_pred_labels=text_pred_labels, word_ids=word_ids)
        pred_pair_list_index_batch = []
        pred_pair_list_index_single = []

        for i in pred_pair_list:
            for j in i:
                pred_pair_list_index_single.append(list(range(j[0], j[1] + 1)) + [j[2]])
            pred_pair_list_index_batch.append(pred_pair_list_index_single)
            pred_pair_list_index_single = []

        pred_pair_list_index_batch, gold_sentiments = self.separate_tensors(data=pred_pair_list_index_batch)

        original_sentence = []

        for i in input_ids:
            original_sentence.append(self.tokenizer_large.tokenize(self.tokenizer_large.decode(i)))

        positions_single = []
        positions_batch = []

        for index, spans in enumerate(pred_pair_list_index_batch):
            for single_span in spans:
                positions_single.append(
                    [i for i, value in enumerate(word_ids[index]) if value in single_span])
            positions_batch.append(positions_single)
            positions_single = []

        aspect_term = []
        selected_data = []

        for positions, sentence in zip(positions_batch, original_sentence):
            for position in positions:
                selected_data.append([sentence[i] for i in position])
            aspect_term.append(selected_data)
            selected_data = []

        aspects = self.merge_tokens_three(nested_list=copy.deepcopy(aspect_term))

        return aspects, pred_pair_list_index_batch, gold_sentiments

    def to_tensor(self, output_dict):
        for i, item in enumerate(output_dict["gold_pair_list_index_batch"]):
            for j, inner_item in enumerate(item):
                output_dict["gold_pair_list_index_batch"][i][j] = torch.tensor(inner_item)

    def padding_and_truncation(self, output_tensor):
        gold_pair_list_index_batch = output_tensor["gold_pair_list_index_batch"]

        big_tensor_two_format = self.get_big_tensor_two_format(tensor_groups=gold_pair_list_index_batch,
                                                               max_length=self.aspect_term_max_length,
                                                               max_num=self.aspect_term_num)

        output_tensor["gold_pair_list_index_batch"] = big_tensor_two_format

    def get_big_tensor_two_format(self, tensor_groups, max_length, max_num):
        processed_group_tensors = []

        for group in tensor_groups:
            new_tensor = torch.full((max_num, max_length), -200)

            for index, tensor in enumerate(group):
                if index < max_num:
                    if tensor.size(0) < max_length:
                        padding_size = max_length - tensor.size(0)
                        padded_tensor = F.pad(tensor, (0, padding_size), value=-200)
                    elif tensor.size(0) > max_length:
                        padded_tensor = tensor[:max_length]
                    else:
                        padded_tensor = tensor

                    new_tensor[index] = padded_tensor

            processed_group_tensors.append(new_tensor)

        big_tensor_group = torch.stack(processed_group_tensors)
        return big_tensor_group

    def separate_tensors(self, data):
        values_without_tensors = []
        tensors = []

        for sublist in data:
            current_sublist = []
            current_tensors = []
            for item in sublist:
                current_sublist.append(item[:-1])
                current_tensors.append(item[-1])
            values_without_tensors.append(current_sublist)
            tensors.append(current_tensors)

        return values_without_tensors, tensors

    def generate_count_list(self, lst):
        count_dict = {}
        result = []
        for item in lst:
            element = tuple(item)
            if element in count_dict:
                count_dict[element] += 1
            else:
                count_dict[element] = 1
            result.append(count_dict[element])
        return result

    def replace_nth_occurrence(self, s, target, replacement, n):
        index = -1
        for _ in range(n):
            index = s.find(target, index + 1)
            if index == -1:
                return s
        return s[:index] + replacement + s[index + len(target):]
