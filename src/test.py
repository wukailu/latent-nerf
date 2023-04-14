# In[]:
my_dict = {'id': [0, 1, 2],
           'name': ['mary', 'bob', 'eve'],
           'age': [24, 53, 19]}
from datasets import Dataset
dataset = Dataset.from_dict(my_dict)
dataset.save_to_disk("/data/tmp/mem_data")

dataset = Dataset.load_from_disk("/data/tmp/mem_data")

