import json
# from typing import List for python < 3.9
from datetime import date, datetime
from utils import Post_Train_Techniques, Language, License, Backbone, enum_to_json, json_to_enum

# instantiate Enum objects

SFT = Post_Train_Techniques.SFT
Continuous_Pretraining = Post_Train_Techniques.Continuous_Pretraining
RLHF = Post_Train_Techniques.RLHF
DPO = Post_Train_Techniques.DPO
Merge = Post_Train_Techniques.Merge

English = Language.English
Chinese = Language.Chinese
Japanese = Language.Japanese
Spanish = Language.Spanish
French = Language.French
Italian = Language.Italian
German = Language.German

mit = License.mit
apache_2_0 = License.apache_2_0
llama2 = License.llama2
cc_by_sa_3_0 = License.cc_by_sa_3_0

Llama_2_7B = Backbone.Llama_2_7B
Mixtral_7B = Backbone.Mixtral_7B

class Dataset:
    def __init__(self, name: str, urls: list[str], license: License) -> None:
        self.name = name
        self.urls = urls
        self.license = license

        self.token_size = None
        self.storage_size = None
    def __str__(self) -> str:
        return f"name: {self.name}, urls: {self.urls}, license: {self.license}."
    def add_size(self, tokens: int = 0, storage: int = 0): # dataset size in #tokens and storage (in MB)
        self.token_size = tokens
        self.storage_size = storage
    def to_dict(self):
        return {
            'name': self.name,
            'urls': self.urls,
            'license': enum_to_json(self.license),
            'token_size': self.token_size,
            'storage_size': self.storage_size
        }
    @classmethod
    def from_dict(cls, data):
        instance = cls(
            data['name'],
            data['urls'],
            json_to_enum(License, data['license'])
        )
        # use .get() to avoid KeyError 
        instance.token_size = data.get('token_size')
        instance.storage_size = data.get('storage_size')
        return instance

class LLM:
    def __init__(self, name: str, urls: list[str], num_param: int, context_window: int, backbone: Backbone, license: License) -> None: # essentials
        self.name = name # use huggingface checkpoint whenever possible
        self.urls = urls
        self.num_param = num_param # in billion
        self.context_window = context_window # in #tokens
        self.backbone = backbone
        self.license = license

        self.easy_case = []
        self.medium_case = []
        self.hard_case = []
        self.paper_link = None
        self.pretrained_datasets = None
        self.languages = None
        self.post_train_techniques = None
        self.post_train_datasets = None
        self.release_date = None
        self.arena_rank, self.arena_elo, self.arena_votes = None, None, None
        self.pretraining_cost, self.post_training_cost = None, None

    def __str__(self) -> str:
        return f"name: {self.name}, urls: {self.urls}, num_param: {self.num_param}, context_window: {self.context_window}, backbone: {self.backbone.name}, license: {self.license.name}."
    def update_paper_link(self, link: str) -> None:
        self.paper_link = link
    def update_pretrained_datasets(self, datasets: list[Dataset]) -> None:
        self.pretrained_datasets = datasets
    def update_languages(self, languages: list[Language]) -> None:
        self.languages = languages
    def update_post_train_techniques(self, techniques: list[Post_Train_Techniques]) -> None:
        self.post_train_techniques = techniques
    def update_post_train_datasets(self, datasets: list[Dataset]) -> None:
        self.post_train_datasets = datasets
    def update_release_date(self, release_date: date) -> None:
        self.release_date = release_date
    def update_chatbot_arena_info(self, rank: int, Elo: int, votes: int) -> None:
        self.arena_rank = rank
        self.arena_elo = Elo
        self.arena_votes = votes
    def upload_easy_case(self, input: str, output: str) -> None:
        self.easy_case.append((input, output))
    def upload_medium_case(self, input: str, output: str) -> None:
        self.medium_case.append((input, output))
    def upload_hard_case(self, input: str, output: str) -> None:
        self.hard_case.append((input, output))
    def update_pretraining_cost(self, cost: int) -> None: # cost in usd
        self.pretraining_cost = cost
    def update_post_training_cost(self, cost: int) -> None: # cost in usd
        self.post_training_cost = cost
    def to_dict(self):
        return {
            'name': self.name,
            'urls': self.urls,
            'num_param': self.num_param,
            'context_window': self.context_window,
            'backbone': enum_to_json(self.backbone),
            'license': enum_to_json(self.license),

            'easy_case': self.easy_case,
            'medium_case': self.medium_case,
            'hard_case': self.hard_case,
            'paper_link': self.paper_link,
            # need to convert
            'pretrained_datasets': [dataset.to_dict() for dataset in self.pretrained_datasets] if self.pretrained_datasets else None,
            'languages': [enum_to_json(lang) for lang in self.languages] if self.languages else None,
            'post_train_techniques': [enum_to_json(tech) for tech in self.post_train_techniques] if self.post_train_techniques else None,
            'post_train_datasets': [dataset.to_dict() for dataset in self.post_train_datasets] if self.post_train_datasets else None,

            'release_date': self.release_date,
            'arena_rank': self.arena_rank, 'arena_elo': self.arena_elo, 'arena_votes': self.arena_votes,
            'pretraining_cost': self.pretraining_cost, 'post_training_cost': self.post_training_cost
        }
    @classmethod
    def from_dict(cls, data):
        instance = cls(
            data['name'],
            data['urls'],
            data['num_param'],
            data['context_window'],
            json_to_enum(Backbone, data['backbone']),
            json_to_enum(License, data['license'])
        )
        # use .get() to avoid KeyError
        instance.easy_case = data.get('easy_case', [])
        instance.medium_case = data.get('medium_case', [])
        instance.hard_case = data.get('hard_case', [])
        instance.paper_link = data.get('paper_link')
        instance.pretrained_datasets = [Dataset.from_dict(d) for d in data.get('pretrained_datasets', [])]
        instance.languages = [json_to_enum(Language, lang) for lang in data.get('languages', [])]
        instance.post_train_techniques = [json_to_enum(Post_Train_Techniques, tech) for tech in data.get('post_train_techniques', [])]
        instance.post_train_datasets = [Dataset.from_dict(d) for d in data.get('post_train_datasets', [])]
        if 'release_date' in data and data['release_date']:
            instance.release_date = datetime.strptime(data['release_date'], '%Y-%m-%d').date()
        instance.arena_rank = data.get('arena_rank')
        instance.arena_elo = data.get('arena_elo')
        instance.arena_votes = data.get('arena_votes')
        instance.pretraining_cost = data.get('pretraining_cost')
        instance.post_training_cost = data.get('post_training_cost')
        return instance



if __name__ == "__main__":
    llm = LLM("meta-llama/Llama-2-7b", "https://huggingface.co/meta-llama/Llama-2-7b", 7, 4000, Llama_2_7B, llama2)
    # write to json file
    with open('llm.json', 'w') as json_file:
        json.dump(llm.to_dict(), json_file, indent=4)
    
    dset = Dataset("wikipedia", "https://huggingface.co/datasets/wikipedia", cc_by_sa_3_0)
    # write to json file
    with open('dset.json', 'w') as json_file:
        json.dump(llm.to_dict(), json_file, indent=4)