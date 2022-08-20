
if __name__ == '__main__':
  conn,cur = connect('db/ocr_db')
  dataframe = get_dataset("/content/drive/MyDrive/datasets/predict_images/raw_images/reports/dataset.csv",conn,"rawdata")
  create_load(cur,conn,"rawdata")
  fetch_data(cur,"rawdata")



