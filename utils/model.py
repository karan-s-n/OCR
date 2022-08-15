from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import utils.DatasetLoader as DL
from utils import utils as ut
from utils import imageutils as im
import imp
import pandas as pd
print(os.getcwd())
np.random.seed(42)
tf.random.set_seed(42)

class OCRModel:
  def __init__(self,base_path,pretrained=False):
    if pretrained:
      pass
    else:
      dataprep = DL.DataLoad(base_path)
      dataprep.get_word_list()
      #split of data into train , test , valid
      self.train_data , self.test_data,self.valid_data = dataprep.train_valid_test_split() 
  
  def data_split(self):
    pass


if __name__ == '__main__':
  print("Loading OCR model")
  ocr = OCRModel()
  ocr.data_split()







