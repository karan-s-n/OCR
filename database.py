import pandas as pd
import sqlite3

def connect(dbname):
  conn = sqlite3.connect(dbname)
  cur = conn.cursor()
  return conn,cur

def get_dataset(path,conn,tablename):
  df = pd.read_csv(path)
  df.to_sql(tablename, conn, if_exists='replace', index = False)
  return df

def create_load(cur,conn,tablename):
  cur.execute(f'CREATE TABLE IF NOT EXISTS {tablename} (imagpath text, x number,y number,w number,h number , wordpath text , predicttext text)')
  conn.commit()

def fetch_data(cur,table):
  cur.execute(f''' SELECT * FROM {table}''')
  for row in cur.fetchall():
      print (row)
      break

if __name__ == '__main__':
  conn,cur = connect('db/ocr_db')
  dataframe = get_dataset("/content/drive/MyDrive/datasets/predict_images/raw_images/reports/dataset.csv",conn,"rawdata")
  create_load(cur,conn,"rawdata")
  fetch_data(cur,"rawdata")



