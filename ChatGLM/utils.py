import json
from sklearn import metrics
import numpy as np
from torch.utils.data import Dataset
import random
import numpy as np
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.utils import PaddingStrategy
from imblearn.under_sampling import RandomUnderSampler
from unidecode import unidecode
from collections import defaultdict


def cleanName(dirtyName):
    name = dirtyName.lower()
    name = name.replace('\xa0', ' ')
    name = name.replace('.', ' ')
    name = name.replace('dr.', '')
    name = name.replace('dr ', '')
    name = name.replace(' 0001', '')
    temp = name.split(' ')
    if len(temp) == 2:
        if '-' in temp[1] and '-' not in temp[0]:
            a = temp[1]
            temp[1] = temp[0]
            temp[0] = a
    k = []
    for t in temp:
        if t != '' and t != ' ':
            k.append(t)
    name = '_'.join(k)
    name = name.replace('-', '')
    return name

def clean_name(name):
    # print(name)
    name = unidecode(name)
    name = name.lower()
    new_name = ""
    for a in name:
        if a.isalpha():
            new_name += a
        else:
            new_name = new_name.strip()
            new_name += " "
    return new_name.strip()


def co_cal(core_name, paper1, paper_list):
    core_name = clean_name(core_name)
    auther_list_1 = [clean_name(paper1["authors"][ins_index]["name"]).strip() for ins_index in
                     range(len(paper1["authors"]))]
    org_1 = ' '.join([i['org'] for i in paper1['authors'] if i['org'] != '']).split()
    n1_venue = paper1['venue'].split()
    n1_keyword = set((' '.join(paper1['keywords']).split()))
    # n1_title = set((' '.join(paper1['title']).split()))

    if core_name in auther_list_1:
        auther_list_1.remove(core_name)

    num = 0
    num_org = 0
    num_venue = 0
    num_keyword = 0
    num_title = 0
    for paper2 in paper_list:
        if paper1['id'] != paper2['id']:
            auther_list_2 = [clean_name(paper2["authors"][ins_index]["name"]).strip() for ins_index in
                             range(len(paper2["authors"]))]
            org_2 = ' '.join([i['org'] for i in paper2['authors'] if i['org'] != '']).split()
            n2_venue = paper2['venue'].split()
            n2_keyword = set((' '.join(paper2['keywords']).split()))
            # n2_title = set((' '.join(paper2['title']).split()))

            # 共同作者
            if core_name in auther_list_2:
                auther_list_2.remove(core_name)
            if len(set(auther_list_1) & set(auther_list_2)) > 0:
                num += 1
            # 共同的机构
            if len(set(org_1) & set(org_2)) > 0:
                num_org += 1
            if len(set(n1_venue) & set(n2_venue)) > 0:
                num_venue += 1
            if len(set(n1_keyword) & set(n2_keyword)) > 0:
                num_keyword += 1
            # if len(set(n1_title) & set(n2_title))>0:
            #     num_title+=1

    return [round((num + 1) / len(paper_list),3),
            round((num_org + 1) / len(paper_list),3),
            round((num_venue + 1) / len(paper_list),3),
            round((num_keyword + 1) / len(paper_list),3)]


def get_year(profile_list):
    all_year = defaultdict(int)
    for paper in profile_list:
        all_year[paper.get("year", 0)] += 1
    return all_year


def get_auther_name(auther_list):
    auther_dict = defaultdict(int)
    for one_paper_auther in auther_list:
        for auther in one_paper_auther:
            if auther["name"] != '':
                auther_dict[cleanName(auther["name"])]+=1

    # 对关键字按出现次数降序排序
    sorted_auther = sorted(auther_dict.items(), key=lambda x: x[1], reverse=True)

    # 输出前50个关键字
    top_100_auther = [auther_name for auther_name, _ in sorted_auther[:100]]

    return top_100_auther

def cal_pre_keywords(keywords_dict):
    keyword_dict = {}
    for paper_keywords in  keywords_dict:
        for keyword in paper_keywords:
            if keyword in keyword_dict:
                keyword_dict[keyword]+=1
            else:
                keyword_dict[keyword]=1

    # 对关键字按出现次数降序排序
    sorted_keywords = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)

    # 输出前50个关键字
    top_200_keywords = [keyword for keyword, _ in sorted_keywords[:200]]

    return top_200_keywords


class INDDataSet(Dataset):
    '''
        iteratively return the profile of each author 
    '''
    def __init__(self, dataset, tokenizer, max_source_length, max_target_length,sim_feature):
        super(INDDataSet, self).__init__()
        self.author, self.pub = dataset  
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.feat_sim = sim_feature
        author_keys = self.author.keys()
        train_keys = []
        labels = []
        for key in author_keys :
            for i in self.author[key]['outliers']:
                train_keys.append({
                    "pub": i,
                    "author": key,
                    "label": 0
                }) 
                labels.append(0)
            for i in self.author[key]['normal_data']:
                train_keys.append({
                    "pub": i,
                    "author": key,
                    "label": 1
                })
                labels.append(1)
        rus = RandomUnderSampler(random_state=0)
        keys_ids = list(range(0,len(train_keys)))
        keys_ids = [ [x, 0] for x in keys_ids ]
        sampled_keys,_ = rus.fit_resample(keys_ids, labels)   # 根据labels随机取train_keys，取出来的label中0和1一样多
        self.train_keys = [train_keys[i[0]] for i in sampled_keys]
        random.shuffle(self.train_keys)
        # origin
        self.instruct = "Identify the abnormal text from the text collection according to the following rules: Here is a collection of paper titles: ### {} ###. Does the paper title ### {} ### belong to the main part of these papers, give me an answer between 'yes' or 'no'."

        # "gpt3.5 optimize"
        # self.instruct = "Analyze a collection of paper titles and their venue to identify abnormal text. The text collection consists of paper titles: ### {} ###, along with the venue: ### {} ###,. Determine if the paper title ### {} ### with the venue ### {} ### belongs to the main body of these papers. Provide a binary answer: 'yes' if it belongs, or 'no' if it does not."

        # self.instruct= "This is a name disambiguation task. Below are all the papers authored by {}. However, some of these papers might belong to another person with the same name. The following information includes all of the author's papers, including all paper titles: ### {} ###, all venues: ### {} ###, authors: ### {} ###, keywords: ### {} ###, and the publication years: #{}#. Determine if the paper titled ### {} ###, published in the venue ### {} ###, authored by ### {} ###, with keywords ### {} ###, in the year #{}# belongs to this author. Provide a binary answer: 'yes' if it belongs, or 'no' if it does not."
        self.yes_token = self.tokenizer.encode(text='yes', add_special_tokens=False, truncation=True,)
        self.no_token = self.tokenizer.encode(text='no', add_special_tokens=False, truncation=True,)

    def __len__(self):
        return len(self.train_keys)

    def __getitem__(self, index):
        profile = self.author[self.train_keys[index]['author']]['normal_data'] +self.author[self.train_keys[index]['author']]['outliers']

        auther = self.author[self.train_keys[index]['author']]['name']
        # sim_feature = self.feat_sim[self.train_keys[index]['author']][self.train_keys[index]['pub']]
        profile_list = [self.pub[p] for p in profile if p != self.train_keys[index]['pub']]

        pro_venue = [self.pub[p]['venue'] for p in profile if p != self.train_keys[index]['pub']]
        profile = [self.pub[p]['title'] for p in profile if p != self.train_keys[index]['pub']] #delete disambiguate paper

        # co_auther, co_auther_org, co_venu, co_keywords
        # colist = co_cal(self.author[self.train_keys[index]['author']]['name'], self.pub[self.train_keys[index]['pub']], profile_list)

        random.shuffle(profile)
        random.shuffle(pro_venue)

        # breakpoint()
        # limit context token lenth up to max_len - 700

        # title
        tokenized_profile = [self.tokenizer.tokenize(i) for i in profile]
        len_profile = [len(i) for i in tokenized_profile]
        sum_len = sum(len_profile)
        if sum_len> self.max_source_length-500: # left 500 for the instruction templete
            total_len = 0
            p = 0   
            while total_len < self.max_source_length-500 and p < sum_len:
                total_len += len_profile[p]
                p += 1
            profile = profile[:p-1]  # 如果长度不够，不取所有论文标题作为检查

        # venue
        tokenized_venue = [self.tokenizer.tokenize(i) for i in pro_venue]
        len_venue = [len(i) for i in tokenized_venue]

        sum_len = sum(len_venue)
        if sum_len > 2500:
            total_len = 0
            p = 0
            while total_len < 2500 and p < sum_len:
                total_len += len_venue[p]
                p += 1
            pro_venue = pro_venue[:p - 1]

        profile_text = ' # '.join(profile)
        venue_text = ' # '.join(pro_venue)

        title = self.pub[self.train_keys[index]['pub']]['title']
        title = title if len(self.tokenizer.tokenize(title))<200 else ' '.join(title.split(' ')[:100]) #limit the disambiguate paper title token lenth

        venue = self.pub[self.train_keys[index]['pub']]['venue']
        venue = venue if len(self.tokenizer.tokenize(venue)) < 100 else ' '.join(venue.split(' ')[:50])  # limit the disambiguate paper title token lenth

        # context = self.instruct.format(profile_text, venue_text,title,venue)
        context = self.instruct.format(profile_text,title)

        input_ids = self.tokenizer.encode(text=context, add_special_tokens=True, truncation=True, max_length=self.max_source_length)
        label_ids = self.yes_token if self.train_keys[index]['label'] else self.no_token
        input_ids = input_ids + label_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * (len(input_ids)-2) + label_ids + [self.tokenizer.eos_token_id]  # -2是为了和input_ids长度一样

        return {
            "input_ids":input_ids,   # 问题(包含该作者的所有论文名字)
            "labels":labels,           # 答案
            "author":self.train_keys[index]['author'], # 作者id
            "pub":self.train_keys[index]['pub'],    # 检查的这篇论文的id
        }

@dataclass
class DataCollatorForIND:
    """
        borrow and modified from transformers.DataCollatorForSeq2Seq
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        # breakpoint()
        features = self.tokenizer.pad(
            features,
            padding=True,
            max_length=max_label_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        # breakpoint() # [(len(features[i]['input_ids']),len(features[i]['labels'])) for i in range(4)]
        return features    

class IND4EVAL(Dataset):
    def __init__(self, dataset, tokenizer, max_source_length, max_target_length,sim_feature):
        super(IND4EVAL, self).__init__()
        self.author, self.pub = dataset  
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.feat_sim = sim_feature
        author_keys = self.author.keys()

        self.val_set = []
        if 'normal_data' in self.author[list(author_keys)[0]]:
            for key in author_keys:   
                for pub_key in self.author[key]['normal_data']:   
                    self.val_set.append({
                        'pub':pub_key,
                        'author':key,
                        'label':1
                    }) 
                for pub_key in self.author[key]['outliers']:
                    self.val_set.append({
                        'pub':pub_key,
                        'author':key,
                        'label':0
                    }) 
        elif 'papers' in self.author[list(author_keys)[0]]:
            for key in author_keys:   
                for pub_key in self.author[key]['papers']:   
                    self.val_set.append({
                        'pub':pub_key,
                        'author':key,
                    })
        # origin
        self.instruct = "Identify the abnormal text from the text collection according to the following rules: Here is a collection of paper titles: ### {} ###. Does the paper title ### {} ### belong to the main part of these papers, give me an answer between 'yes' or 'no'."
        # now
        # self.instruct = "Analyze a collection of paper titles and their venue to identify abnormal text. The text collection consists of paper titles: ### {} ###, along with the venue: ### {} ###. Determine if the paper title ### {} ### with the venue ### {} ### belongs to the main body of these papers. Provide a binary answer: 'yes' if it belongs, or 'no' if it does not."
        # self.instruct = "This is a name disambiguation task. Below are all the papers authored by {}. However, some of these papers might belong to another person with the same name. The following information includes all of the author's papers, including all paper titles: ### {} ###, all venues: ### {} ###, authors: ### {} ###, keywords: ### {} ###, and the publication years: #{}#. Determine if the paper titled ### {} ###, published in the venue ### {} ###, authored by ### {} ###, with keywords ### {} ###, in the year #{}# belongs to this author. Provide a binary answer: 'yes' if it belongs, or 'no' if it does not."

    def __len__(self):
        return len(self.val_set)
    
    def __getitem__(self, index):
        if "normal_data" in self.author[self.val_set[index]['author']]:
            profile = self.author[self.val_set[index]['author']]['normal_data'] +self.author[self.val_set[index]['author']]['outliers']
        elif "papers" in self.author[self.val_set[index]['author']]:
            profile = self.author[self.val_set[index]['author']]['papers']
        else:
            raise("No profile found")

        pro_venue = [self.pub[p]['venue'] for p in profile if p != self.val_set[index]['pub']]

        profile = [self.pub[p]['title'] for p in profile if p != self.val_set[index]['pub']] #delete disambiguate paper

        random.shuffle(profile)
        random.shuffle(pro_venue)

        tokenized_profile = [self.tokenizer.tokenize(i) for i in profile]
        len_profile = [len(i) for i in tokenized_profile]
        sum_len = sum(len_profile)
        if sum_len> self.max_source_length-500:
            total_len = 0
            p = 0   
            while total_len < self.max_source_length-500 and p < sum_len:
                total_len += len_profile[p]
                p += 1
            profile = profile[:p-1]

        tokenized_venue = [self.tokenizer.tokenize(i) for i in pro_venue]
        len_venue = [len(i) for i in tokenized_venue]
        sum_len = sum(len_venue)
        if sum_len > 2500:
            total_len = 0
            p = 0
            while total_len < 2500 and p < sum_len:
                total_len += len_venue[p]
                p += 1
            pro_venue = pro_venue[:p - 1]


        profile_text = ' # '.join(profile)
        venue_text = ' # '.join(pro_venue)


        title = self.pub[self.val_set[index]['pub']]['title']
        title = title if len(self.tokenizer.tokenize(title))<200 else ' '.join(title.split(' ')[:100])

        venue = self.pub[self.val_set[index]['pub']]['venue']
        venue = venue if len(self.tokenizer.tokenize(venue)) < 100 else ' '.join(venue.split(' ')[:100])


        # context = self.instruct.format(auther, profile_text, venue_text, pro_auther, keyword_text, years, title, venue,
        #                                auther_name, this_paper_keywords, year)
        # context = self.instruct.format(profile_text, venue_text, title, venue)
        context = self.instruct.format(profile_text, title)

        return {
            "input_ids":context,
            "author":self.val_set[index]['author'],
            "pub":self.val_set[index]['pub'],
        }
