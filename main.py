from tensorflow import keras
from database import *
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import re
import cv2
import numpy as np
import utils as ut
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
np.random.seed(42)
tf.random.set_seed(42)
import json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
os.environ['USE_TORCH'] = '1'
model = ocr_predictor(pretrained=True)
import pandas as pd


def format_to_csv(result_df,image_path):
    image = cv2.imread(image_path,0)
    h,w = image.shape
    data_df = pd.DataFrame()
    for i in range(len(result_df)):
        block_df = normalize(result_df["pages.blocks"][i])
        for j in block_df.index:
                line_df = normalize(block_df["lines"][j])
                for k in line_df.index:
                    word_df = normalize(line_df["words"][k])
                    word_df["pages.page_idx"]=i
                    word_df["blocks.block_idx"]=j
                    word_df["line.line_idx"]=k
                    word_df["word.line_idx"]=word_df.index.copy()
                    for w in range(0,len(word_df)):
                        word = word_df.loc[w,"geometry"]
                        word_df.loc[w,"x1"]=   int(word[0][0]*w)
                        word_df.loc[w,"y1"]=   int(word[0][1]*h)
                        word_df.loc[w,"x2"]=   int(word[1][0]*w)
                        word_df.loc[w,"y2"]=   int(word[1][1]*h)
                    data_df = pd.concat([data_df,word_df])
    return data_df

def normalize(data):
    df = pd.DataFrame(data)
    json_struct = json.loads(df.to_json(orient="records"))
    result = pd.json_normalize(json_struct) #use pd.io.json
    result=result.reset_index()
    return result

def resize(image_list):
        global max_height
        global max_width
        max_height = max(list(map(lambda x :x.shape[0], image_list)))
        max_width = max(list(map(lambda x :x.shape[1], image_list)))
        dim = (max_width,max_height)
        return [cv2.resize(img, dim, interpolation = cv2.INTER_AREA) for img in image_list]
def get_data(string,pattern,df,key):
    try:
        field_raw = df[df[key].str.contains(string)][key].values[0]
        field = re.findall(pattern,field_raw)[0]
        return field
    except:
        return ""

def call_OCR(image_path,filename):
    import json
    # PDF
    doc = DocumentFile.from_images(image_path)
    # Analyze
    result = model(doc)
    json_output = result.export()
    if len(json_output)!=0:
        result_df = pd.DataFrame(json_output)
        json_struct = json.loads(result_df.to_json(orient="records"))
        result_df = pd.json_normalize(json_struct) #use pd.io.json
        result_df=result_df[["pages.page_idx","pages.blocks"]]
        data_df = format_to_csv(result_df,image_path)
        blocks_data = pd.DataFrame(columns = ["text"],index = list(set(data_df["blocks.block_idx"].values)))
        for blockidx in blocks_data.index:
            filter_data = data_df[data_df["blocks.block_idx"]==blockidx]["value"]
            blocks_data.loc[blockidx,filename] =" ".join(filter_data.values)
    return blocks_data


############################### IMPORTED ALL THE PACKAGES ######################################
class ImagePreprocessing:
  def __init__(self):
    self.batch_size = 64
    self.padding_token = 99
    self.image_width = 128
    self.image_height = 32
    self.AUTOTUNE = tf.data.AUTOTUNE


  def preprocess_png_image(self,image_path):
      img_size=(self.image_width, self.image_height)
      image = tf.io.read_file(image_path)
      image = tf.image.decode_png(image, 1)
      image = self.distortion_free_resize(image, img_size)
      image = tf.cast(image, tf.float32) / 255.0
      return image


  def preprocess_jpeg_image(self,image_path):
      img_size=(self.image_width, self.image_height)
      image = tf.io.read_file(image_path)
      image = tf.image.decode_jpeg(image, 1)
      image = self.distortion_free_resize(image, img_size)
      image = tf.cast(image, tf.float32) / 255.0
      return image


  def vectorize_label(self,label):
      label =  self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
      length = tf.shape(label)[0]
      pad_amount = self.max_len - length
      label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=self.padding_token)
      return label


  def process_images_labels(self,image_path, label):
      image = self.preprocess_jpeg_image(image_path)
      label = self.vectorize_label(label)
      return {"image": image, "label": label}


  def prepare_dataset(self,image_paths, labels,type_data):
      print("PREPARING DATASET: ",type_data)
      dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
self.process_images_labels, num_parallel_calls=self.AUTOTUNE
      )
      return dataset.batch(self.batch_size).cache().prefetch(self.AUTOTUNE)



  def distortion_free_resize(self,image, img_size):
      w, h = img_size
      image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

      # Check tha amount of padding needed to be done.
      if h>tf.shape(image)[0]:
        pad_height = h - tf.shape(image)[0]
      else:
        pad_height = 0
      if w > tf.shape(image)[1]:
        pad_width = w - tf.shape(image)[1]
      else:
        pad_width=0



      # Only necessary if you want to do same amount of padding on both sides.
      if pad_height % 2 != 0:
          height = pad_height // 2
          pad_height_top = height + 1
          pad_height_bottom = height
      else:
          pad_height_top = pad_height_bottom = pad_height // 2

      if pad_width % 2 != 0:
          width = pad_width // 2
          pad_width_left = width + 1
          pad_width_right = width
      else:
          pad_width_left = pad_width_right = pad_width // 2

      image = tf.pad(
          image,
          paddings=[
              [pad_height_top, pad_height_bottom],
              [pad_width_left, pad_width_right],
              [0, 0],
          ],
      )

      image = tf.transpose(image, perm=[1, 0, 2])
      image = tf.image.flip_left_right(image)
      return image

  def padding(self):
    for data in self.train_ds.take(1):
        images, labels = data["image"], data["label"]

        _, ax = plt.subplots(4, 4, figsize=(15, 8))

    for i in range(16):
        img = images[i]
        img = tf.image.flip_left_right(img)
        img = tf.transpose(img, perm=[1, 0, 2])
        img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]
        # Gather indices where label!= padding_token.
        label = labels[i]
        indices = tf.gather(label, tf.where(tf.math.not_equal(label, self.padding_token)))
        # Convert to string.
        label = tf.strings.reduce_join(self.num_to_char(indices))
        label = label.numpy().decode("utf-8")

        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")
    #plt.show()



class OCRModel(ImagePreprocessing):
  def __init__(self,training_path,untar_iamdataset=False):
    print("###STEP1: DATA LOADING")
    super().__init__()
    self.base_path = training_path
    self.base_image_path = os.path.join(self.base_path , "IAM_Words","data","words")
    self.mnist_path = os.path.join(self.base_path , "mnist_Words","labels.json")
    self.mnist_image_path = os.path.join(self.base_path , "mnist_Words","dataset","v011_words_small")
    if untar_iamdataset:
        ut.extract_iamdataset_file(self.base_image_path+"/words.tgz",extract_path=self.base_image_path)
    self.dataprep = ut.DataLoad(self.base_path)
    self.dataprep.get_word_list()
    self.train_labels_cleaned = []
    self.characters = set()
    self.max_len = 0
    self.mnist_paths,self.mnist_labels = ut.get_mnist_dataset(self.mnist_path,self.mnist_image_path)
    self.train_data , self.test_data,self.valid_data = self.dataprep.train_valid_test_split()

  def prepare_data(self):
      print("##STEP2: DATA SPLITING")
      #get images path and labels 
      self.tr_images, self.tr_labels = ut.get_image_paths_and_labels(self.train_data,self.base_image_path)
      self.valid_images,self.valid_labels = ut.get_image_paths_and_labels(self.test_data,self.base_image_path)
      self.test_images, self.test_labels = ut.get_image_paths_and_labels(self.valid_data,self.base_image_path)


  def clean_labels(self,labels):
    cleaned_labels = []
    for label in labels:
        label = label.split(" ")[-1].strip()
        cleaned_labels.append(label)
    return cleaned_labels

  def data_clean(self):
    print("###STEP3: DATA CLEANING")
    # Find maximum length and the size of the vocabulary in the training data.
    self.train_labels_cleaned = []
    self.characters = set()
    self.max_len = 0

    for label in self.tr_labels:
        label = label.split(" ")[-1].strip()
        for char in label:
            self.characters.add(char)

        self.max_len = max(self.max_len, len(label))
        self.train_labels_cleaned.append(label)

    self.characters = sorted(list(self.characters))
    with open("characters.txt","w") as f:
      f.write(",".join(self.characters))
    self.char_to_num = StringLookup(vocabulary=list(self.characters), mask_token=None)
    self.num_to_char = StringLookup(vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True)


    self.validation_labels_cleaned = [label.split(" ")[-1].strip() for label in self.valid_labels]
    self.test_labels_cleaned = [label.split(" ")[-1].strip() for label in self.test_labels]

    print("PREPARING DATASET FOR TRAINING")
    #self.tr_images.extend(self.mnist_paths)
    #self.train_labels_cleaned.extend(self.mnist_labels)
    print("NO. of training datasets after extending with mnist", len(self.tr_images))
    self.train_ds = self.prepare_dataset(self.tr_images, self.train_labels_cleaned,"train")
    self.validation_ds = self.prepare_dataset(self.valid_images, self.validation_labels_cleaned,"valid")
    self.test_ds = self.prepare_dataset(self.test_images, self.test_labels_cleaned,"test")  
    print("COMPLETED  DATASET FOR TRAINING")


  def build_model_old(self):
    # Inputs to the model
    input_img = keras.Input(shape=(self.image_width, self.image_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    # First conv block
    x = keras.layers.Conv2D(32,(3,3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv1",)(input_img)
    x = keras.layers.MaxPooling2D((2,2), name="pool1")(x)

    # Second conv block.
    x = keras.layers.Conv2D(64,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv2",)(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    new_shape = ((self.image_width // 4), (self.image_height // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # RNNs.
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.25)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
    )(x)
    x = keras.layers.Dense(
        len(self.char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step.
    output = CTCLossCalculator(name="ctc_loss")(labels, x)

    # Define the model.
    self.model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="OCR"
    )
    # Optimizer.
    opt = keras.optimizers.Adam()
    # Compile the model and return.
    model.compile(optimizer=opt,)
    self.model = model
    return self.model


  def build_model(self):
          input_img = keras.Input(shape=(self.image_width, self.image_height, 1), name="image")
          labels = keras.layers.Input(name="label", shape=(None,))
          x = keras.layers.Conv2D(
              32,
              (3, 3),
              activation="relu",
              kernel_initializer="he_normal",
              padding="same",
              name="Conv1",
          )(input_img)
          x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
          x = keras.layers.Conv2D(
              64,
              (3, 3),
              activation="relu",
              kernel_initializer="he_normal",
              padding="same",
              name="Conv2",
          )(x)
          x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
          new_shape = ((self.image_width // 4), (self.image_height // 4) * 64)
          x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
          x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
          x = keras.layers.Dropout(0.2)(x)

          # RNNs.
          x = keras.layers.Bidirectional(
              keras.layers.LSTM(128, return_sequences=True, dropout=0.25)
          )(x)
          x = keras.layers.Bidirectional(
              keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
          )(x)

          x = keras.layers.Dense(
              len(self.char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2"
          )(x)

          # Add CTC layer for calculating CTC loss at each step.
          output = CTCLayer(name="ctc_loss")(labels, x)

          # Define the model.
          model = keras.models.Model(
              inputs=[input_img, labels], outputs=output, name="handwriting_recognizer"
          )
          # Optimizer.
          opt = keras.optimizers.Adam()
          # Compile the model and return.
          model.compile(optimizer=opt)
          return model


class CTCLossCalculator(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, actual, predict):
        batch_len = tf.cast(tf.shape(actual)[0], dtype="int64")
        input_length = tf.cast(tf.shape(predict)[1], dtype="int64")
        label_length = tf.cast(tf.shape(actual)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(actual, predict, input_length, label_length)
        self.add_loss(loss)
        return predict


def calculate_edit_distance(labels, predictions,max_len):
      saprse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)
      input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
      predictions_decoded = keras.backend.ctc_decode(
            predictions, input_length=input_len, greedy=True
        )[0][0][:, :max_len]
      sparse_predictions = tf.cast(
            tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
        )
      edit_distances = tf.edit_distance(
            sparse_predictions, saprse_labels, normalize=False
        )
      return tf.reduce_mean(edit_distances)

class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model,validation_images,validation_labels,max_len):
        super().__init__()
        self.prediction_model = pred_model
        self.validation_images = validation_images
        self.validation_labels = validation_labels
        self.max_len = max_len

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []

        for i in range(len(self.validation_images)):
            labels = self.validation_labels[i]
            predictions = self.prediction_model.predict(self.validation_images[i])
            edit_distances.append(calculate_edit_distance(labels, predictions,self.max_len).numpy())

        print(
            f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
        )





class Prediction(ImagePreprocessing):

  def __init__(self,pretrained_path,image_path):
    super().__init__()
    self.image_path = image_path
    print("\nSTEP1: LOADING PREDICTION DATASET")
    self.model = keras.models.load_model(pretrained_path+"iamdatasetmodel.pth")
    self.prediction_model = keras.models.Model(self.model.get_layer(name="image").input, self.model.get_layer(name="dense2").output)
    self.max_len = 30
    with open(image_path+"../../characters.txt","r") as f:
      characters = list(f.read())
    AUTOTUNE = tf.data.AUTOTUNE
    self.char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
    self.num_to_char = StringLookup(
        vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

  def load_predict(self):
    self.image_paths = []
    self.image_labels = []
    for label,image in enumerate(self.dataset["word_path"]):
      self.image_paths.append(image)
      self.image_labels.append(str(label))
    print("No. of images present:",len(self.image_paths)) 

  def decode_batch_predictions(self,pred):
      input_len = np.ones(pred.shape[0]) * pred.shape[1]
      results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :self.max_len]
      output_text = []
      for res in results:
          res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
          res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
          output_text.append(res)
      return output_text

  def predict_images(self):
    print("\nSTEP2: PREPARING DATASET FOR PREDICTION")
    self.load_predict()
    self.predict_data = self.prepare_dataset(self.image_paths, self.image_labels,"prediction")
    print("\nSTEP3:  PASSING DATA FOR PREDICTION")
    pred_texts_list=[]
    for batch in self.predict_data:
        batch_images = batch["image"]
        preds = self.prediction_model.predict(batch_images)
        pred_texts = self.decode_batch_predictions(preds)
        pred_texts_list.extend(pred_texts)
    print("Predicted text",len(pred_texts_list))
    print(pred_texts_list)
    self.dataset["Prediction_Text"] = pd.Series(pred_texts_list)
    return pred_texts_list

            
        
if __name__ == '__main__':
  dl = ut.CropImages()
  pretrained_model = dl.pretrained
  crop_words =  dl.crop_words
  pretrained_OPT = dl.pretrained_OPT

  if dl.pretrained_OPT:
      print("Loading pretrained")
      overall_df = {}
      print("iamges path",dl.test_folder + "*.png")
      images_path = [img for img in glob.glob(dl.test_folder + "*.png")]
      print("No. of images loaded",len(images_path))
      filenames = [fstring.rsplit("\\")[-1] for fstring in images_path]
      extract_data = pd.DataFrame(columns=["Company", "Client", "Location", "Date", "Contract"], index=filenames)
      for idx in range(0, len(filenames[0:50])):
          print(filenames[idx])
          final_df = call_OCR(images_path[idx], filenames[idx])
          #print(final_df.head(4))
          overall_df[filenames[idx]] = final_df
      for key, df in overall_df.items():
          df.to_csv(str(key)+".csv")
          Company = "  ".join(df[key][0:8].values)
          if ("Enginee" in Company) or ("Co. Lt" in Company) or ("Eng" in Company):
              Location = get_data("Location", "Location(.*)Client", df, key)
              print("Location", Location)
              Client = get_data("Client", "Client(.*)Method of Borin", df, key)
              print("Client", Client)
              Date = get_data("Date|Dat", "Date(.*)", df, key)
              print("Date", Date)
              Contract = get_data("Contract", "Contract(.*)Location", df, key)
              print("Contract", Contract)
              for col in extract_data.columns:
                  extract_data.loc[key, col] = eval(col)

          elif "Drilling" in Company:
              Location = get_data("Location", "LOCATION(.*)", df, key)
              print("Location", Location)
              Client = get_data("Client", "CLIENT(....)", df, key)
              print("Client", Client)
              Date = get_data("Date|Dat", "DATE(................)", df, key)
              print("Date", Date)
              Contract = get_data("Contract", "JOB NO(.......)", df, key)
              print("Contract", Contract)
              for col in extract_data.columns:
                  extract_data.loc[key, col] = eval(col)

          elif "Dunelm" in Company:
              Location = get_data("Location", "LOCATION(.*)", df, key)
              print("Location", Location)
              Client = get_data("Client", "CLIENT(....)", df, key)
              print("Client", Client)
              Date = get_data("Date|Dat", "Start date(................)", df, key)
              print("Date", Date)
              Contract = get_data("Contract", "Job No(.......)", df, key)
              print("Contract", Contract)
              final_depth = get_data("Final depth", "Job Final depth(....", df, key)
              print("final_depth", final_depth)
              for col in extract_data.columns:
                  extract_data.loc[key, col] = eval(col)


          elif "Soil" in Company:
              Location = get_data("Location", "Location(.*)", df, key)
              print("Location", Location)
              Client = get_data("Client", "Client(.*)", df, key)
              print("Client", Client)
              Date = get_data("Date|Dat", "Date(.*)", df, key)
              print("Date", Date)
              Contract = get_data("Contract", "Contract (.......)", df, key)
              print("Contract", Contract)
              for col in extract_data.columns:
                  extract_data.loc[key, col] = eval(col)
          elif "INVESTIGATION" in Company:
              Location = get_data("Location", "Location(.*)", df, key)
              print("Location", Location)
              Client = get_data("Client", "Client(.*)", df, key)
              print("Client", Client)
              Date = get_data("Date|Dat", "Date(.*)", df, key)
              print("Date", Date)
              Contract = get_data("Contract", "Contract (.......)", df, key)
              print("Contract", Contract)
              for col in extract_data.columns:
                  extract_data.loc[key, col] = eval(col)
          elif "BOREHOLE RECORD" in Company:
              Location = get_data("Location", "Location(.*)", df, key)
              print("Location", Location)
              Client = get_data("Client", "Client(.*)", df, key)
              print("Client", Client)
              Date = get_data("Date|Dat", "Date(.*)", df, key)
              print("Date", Date)
              Contract = get_data("Contract", "Contract(.*)", df, key)
              print("Contract", Contract)
              for col in extract_data.columns:
                  extract_data.loc[key, col] = eval(col)
          elif "BORING METHOD" in "  ".join(df[key][0:20].values):
              Location = get_data("Location", "Location(.*)", df, key)
              print("Location", Location)
              Client = get_data("CLIENT", "CLIENT([a-z|A-Z]*)", df, key)
              print("Client", Client)
              Date = get_data("Date|Dat", "Date(.*)", df, key)
              print("Date", Date)
              Contract = get_data("Contract", "Contract(.*)", df, key)
              print("Contract", Contract)
              for col in extract_data.columns:
                  extract_data.loc[key, col] = eval(col)
      extract_data.to_csv("Extract_data.csv")
      conn,cur = connect('ocr_db.sqlite3')
      dataframe = get_dataset("Extract_data.csv",conn,"Summary")
      summary(cur,conn,"Summary")
      fetch_data(cur,"Summary")
  elif pretrained_model:
    print("PREDICTION PATH:",dl.predict_images)
    pred = Prediction(pretrained_path=dl.pretrainedmodels,image_path = dl.predict_images+dl.predict_words_path)
    if crop_words:
      dl.load_csv()
      dl.get_word_img()
      pred.dataset = dl.load_csv()
      pred.predict_images()
      pred.dataset.to_csv(dl.datasetpath,index=False)
    else:
      pred.dataset = dl.load_csv()
      pred.predict_images()
      pred.dataset.to_csv(dl.datasetpath,index=False)
    conn,cur = connect('ocr_db.sqlite3')
    dataframe = get_dataset(dl.datasetpath,conn,"rawdata")
    create_load_db(cur,conn,"rawdata")
    fetch_data(cur,"rawdata")
    
  else:
    print("TRAINING PATH:",dl.train_images)
    ocr = OCRModel(dl.train_images)
    ocr.prepare_data()
    ocr.data_clean()
    model = ocr.build_model()
    print(model.summary())
    prediction_model = keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)
    edit_distance_callback = EditDistanceCallback(prediction_model)
    edit_distance_callback.validation_ds = ocr.validation_ds
    edit_distance_callback.max_len = ocr.max_len
    history = model.fit(ocr.train_ds, epochs=dl.epochs, callbacks=[edit_distance_callback],)
    print("overall loss",history.history["loss"])
    model.save("iamdatasetmodel.pth")




