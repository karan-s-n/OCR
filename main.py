from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
import utils as ut
import pandas as pd
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

np.random.seed(42)
tf.random.set_seed(42)


class ImagePreprocessing:
  def __init__(self):
    self.batch_size = 64
    self.padding_token = 99
    self.image_width = 128
    self.image_height = 32
    self.AUTOTUNE = tf.data.AUTOTUNE


  def preprocess_image(self,image_path):
      img_size=(self.image_width, self.image_height)
      image = tf.io.read_file(image_path)
      image = tf.image.decode_png(image, 1)
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
      image = self.preprocess_image(image_path)
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
      pad_height = h - tf.shape(image)[0]
      pad_width = w - tf.shape(image)[1]
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
  def __init__(self,training_path):
    print("###STEP1: DATA LOADING")
    super().__init__()
    self.base_path = training_path
    self.base_image_path = os.path.join(self.base_path , "IAM_Words","data","words")
    self.mnist_path = os.path.join(self.base_path , "mnist_Words","labels.json")
    if dl.untar_iamdataset:
        ut.extract_iamdataset_file(self.base_image_path+"/words.tgz",extract_path=self.base_image_path)
    self.dataprep = ut.DataLoad(self.base_path)
    self.dataprep.get_word_list()
    self.train_labels_cleaned = []
    self.characters = set()
    self.max_len = 0
    self.mnist_paths,self.mnist_labels = ut.get_mnist_dataset(self.mnist_path)
    self.train_data , self.test_data,self.valid_data = self.dataprep.train_valid_test_split()

  def prepare_data(self):
      print("##STEP2: DATA SPLITING")
      #get images path and labels 
      self.tr_images, self.tr_labels = ut.get_image_paths_and_labels(self.train_data,self.base_image_path)
      self.valid_images,self.valid_labels = ut.get_image_paths_and_labels(self.test_data,self.base_image_path)
      self.test_images, self.test_labels = ut.get_image_paths_and_labels(self.valid_data,self.base_image_path)
      self.tr_images.extend(self.mnist_paths)
      self.tr_labels.extend(self.mnist_labels)

  
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
    self.train_ds = self.prepare_dataset(self.tr_images, self.train_labels_cleaned,"train")
    self.validation_ds = self.prepare_dataset(self.valid_images, self.validation_labels_cleaned,"valid")
    self.test_ds = self.prepare_dataset(self.test_images, self.test_labels_cleaned,"test")  
    print("COMPLETED  DATASET FOR TRAINING")
  def build_model(self):
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

    # +2 is to account for the two special tokens introduced by the CTC loss.
    # The recommendation comes here: https://git.io/J0eXP.
    x = keras.layers.Dense(
        len(self.char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step.
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model.
    self.model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="handwriting_recognizer"
    )
    # Optimizer.
    opt = keras.optimizers.Adam()
    # Compile the model and return.
    self.model.compile(optimizer=opt)
    return self.model

class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model):
        super().__init__()
        self.prediction_model = pred_model


    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []
        self.validation_images = []
        self.validation_labels = []

        for batch in self.validation_ds:
            self.validation_images.append(batch["image"])
            self.validation_labels.append(batch["label"])
        
        for i in range(len(self.validation_images)):
            labels = self.validation_labels[i]
            predictions = self.prediction_model.predict(self.validation_images[i])
            edit_distances.append(self.calculate_edit_distance(labels, predictions).numpy())

        print(
            f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
        )

    def calculate_edit_distance(self,labels, predictions):
      # Get a single batch and convert its labels to sparse tensors.
      saprse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

      # Make predictions and convert them to sparse tensors.
      input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
      predictions_decoded = keras.backend.ctc_decode(
            predictions, input_length=input_len, greedy=True
        )[0][0][:, :self.max_len]
      sparse_predictions = tf.cast(
            tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
        )

      # Compute individual edit distances and average them out.
      edit_distances = tf.edit_distance(
            sparse_predictions, saprse_labels, normalize=False
        )
      return tf.reduce_mean(edit_distances)


class Prediction(ImagePreprocessing):
  def __init__(self,pretrained_path,image_path):
    super().__init__()
    self.image_path = image_path
    print("\nSTEP1: LOADING PREDICTION DATASET")
    self.model = keras.models.load_model(pretrained_path)
    self.prediction_model = keras.models.Model(self.model.get_layer(name="image").input, self.model.get_layer(name="dense2").output)
    self.max_len = 30
    with open(image_path+"../characters.txt","r") as f:
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
    self.dataset["Prediction_Text"] = pd.Series(pred_texts_list)
    return pred_texts_list

            
        
if __name__ == '__main__':
  dl = ut.CropImages()
  pretrained_model = dl.pretrained
  crop_words =  dl.crop_words
  print("FETCHING MNIST DATASET")

  if pretrained_model:
    print("PREDICTION PATH:",dl.predict_images)
    pred = Prediction(pretrained_path=weights_path,image_path = base_path)
    if crop_words:
      dl.set_attrib()
      dl.load_csv()
      dl.get_word_img()
      pred.dataset = dl.load_csv()
      pred.predict_images()
      pred.dataset.to_csv(dl.datasetpath,index=False)
    else:
      pred.dataset = dl.load_csv()
      pred.predict_images()
      pred.dataset.to_csv(dl.datasetpath,index=False)
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
    history = model.fit( ocr.train_ds, epochs=dl.epochs, callbacks=[edit_distance_callback],)
    model.save(dl.pretrainedmodels+"iamdatasetmodel.pth")
  dbupdate = False
  if dbupdate:
        conn, cur = connect('db/ocr_db')
        dataframe = get_dataset("/content/drive/MyDrive/datasets/predict_images/raw_images/reports/dataset.csv", conn,
                                  "rawdata")
        ut.create_load(cur, conn, "rawdata")
        ut.fetch_data(cur, "rawdata")




