import pandas as pd
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
import tensorflow as tf

class DataLoad:
  def __init__(self,datapath,filter_index = [0,1,8]):
    self.datapath = datapath
    

  def get_word_list(self):
    words_list = []
    words = open(f"{self.datapath}/words.txt", "r").readlines()
    for line in words:
      if line[0] == "#":
          continue
      if line.split(" ")[1] != "err":  # We don't need to deal with errored entries.
          words_list.append(line)
      len(words_list)
    #np.random.shuffle(words_list)
    self.words_list = words_list

  def train_valid_test_split(self):
    words_df = self.words_list
    #spliting 90% of data to train and 5% test and 5% validate
    split_idx = int(0.9 * len(words_df))
    self.train_samples = words_df[:split_idx]
    self.test_samples = words_df[split_idx:]
    val_split_idx = int(0.5 * len(self.test_samples))
    self.validation_samples = self.test_samples[:val_split_idx]
    self.test_samples = self.test_samples[val_split_idx:]
    assert len(words_df) == len(self.train_samples) + len(self.validation_samples) + len(self.test_samples)
    print(f"Total training samples: {len(self.train_samples)}")
    print(f"Total validation samples: {len(self.validation_samples)}")
    print(f"Total test samples: {len(self.test_samples)}")
    return self.train_samples,self.test_samples,self.validation_samples

