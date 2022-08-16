import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
from glob import glob
from tqdm import tqdm


class CropImages:
  def __init__(self):
    with open('config.yml',"r") as f:
        self.config = yaml.load(f, Loader=yaml.Loader)
    self.datasetpath  = self.config["dataset"]
    self.wordspath  = self.config["wordspath"]
    self.report_folder = self.config["report_folder"]
    self.project_path = self.config["project_path"]
    self.epochs = self.config["epochs"]
    self.pretrained = self.config["pretrained"]
    self.crop_words = self.config["crop_words"]
    self.train_data_path = self.config["train_data_path"]
    self.iam_dataset_path = self.config["iam_dataset_path"]
    self.predict_images = self.config["test_folder"]

  def set_attrib(self):
    pass


  def save_words(self,filename,cropped,idx):
        dir = self.wordspath+filename+"/"
        if os.path.isdir(os.path.join(dir)) == False :
          os.makedirs(os.path.join(dir))
          
        filepath = dir+f'_{idx}.jpg'
        cv2.imwrite(filepath, cropped)
        return filepath

  def get_image(self,path):
    filename = path.rsplit("/",2)[-1]
    filename = filename.split(".png")[0]
    image = cv2.imread(path,0)
    return filename , image
  
  def remove_images(self,path):
    images = glob(path)
    for data in images:
        os.remove(data)
  
  def load_csv(self):
    self.dataset = pd.read_csv(self.datasetpath)
    return self.dataset

  def get_word_img(self):
    for image_path in set(self.dataset["image_path"]):
      image_boxes = self.dataset[self.dataset["image_path"]==image_path]
      filename,image = self.get_image(image_path)
      for idx in tqdm(image_boxes.index):
        x,y,w,h = int(image_boxes.loc[idx,"x"]),int(image_boxes.loc[idx,"y"]),int(image_boxes.loc[idx,"w"]),int(image_boxes.loc[idx,"h"])
        cropped_image = image[y:y+h,x:x+w].copy()
        filepath = self.save_words(filename,cropped_image,idx)
        self.dataset.loc[idx,"word_path"] = filepath
    self.dataset.to_csv(self.datasetpath,index=False)
