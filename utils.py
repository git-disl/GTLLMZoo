from enum import Enum, auto

class Post_Train_Techniques(Enum):
    SFT = auto()
    Continuous_Pretraining = auto()
    RLHF = auto()
    DPO = auto()
    Merge = auto()

class Language(Enum):
    English = auto()
    Chinese = auto()
    Japanese = auto()
    Spanish = auto()
    French = auto()
    Italian = auto()
    German = auto()

class License(Enum): # https://huggingface.co/docs/hub/repositories-licenses, substitute - and . to _
    mit = auto()
    apache_2_0 = auto()
    llama2 = auto()
    cc_by_sa_3_0 = auto()

class Backbone(Enum):
    Llama_2_7B = auto()
    Mixtral_7B = auto()

def enum_to_json(enum):
    return enum.name

def json_to_enum(enum_class, name):
    return enum_class[name]

