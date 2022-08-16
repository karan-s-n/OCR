import pandas as pd
import sqlite3

conn = sqlite3.connect('ocr_db')
cur = conn.cursor()

cur.execute('CREATE TABLE IF NOT EXISTS rawdata (imagpath text, x number,y number,w number,h number , wordpath text , predicttext text)')
conn.commit()
df = pd.read_csv("dataset.csv")
df.to_sql('rawdata', conn, if_exists='replace', index = False)
 
cur.execute(''' SELECT * FROM rawdata''')

for row in cur.fetchall():
    print (row)
    break
