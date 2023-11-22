import json

import re
def prepare_data():
    with open("News_Category_Dataset_v2.json") as json_data:
                data_list=[json.loads(line) for line in json_data]
                category_list={}
                vocab={}
                for data in data_list:
                    if data['category'] not in category_list:
                        category_list[data['category']]=len(category_list)+1
                    regex="[!,'.()`?;:]"
                    remove_after_punc=re.sub("[-!,'.()`?;:]","",data["headline"]+" "+data["short_description"])
                    remove_after_punc=remove_after_punc.lower()
                    list_of_word=list(remove_after_punc)
                
                    for headlines in list_of_word:
                        if headlines not in vocab:
                            vocab[headlines]=len(vocab)
                print(len(data_list))
                return vocab,category_list,data_list

