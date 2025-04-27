### 原始两个文件ItemID不对应，需要进行处理
import pandas as pd
df = pd.read_csv('book_info.csv')
df['itemId'] = df['itemId'].astype(str).str.lstrip("0")

# 4. 保存处理后的CSV
df.to_csv('book_info.csv', index=False)

df = pd.read_csv("book_data.csv", header=None)

# 处理第2列（索引为1，因为从0开始）
df[1] = df[1].astype(str).str.lstrip("0")  # 删除前导零

# 保存（不保存列名）
df.to_csv("book_data.csv", header=False, index=False)



## 加载文件 
from sentence_transformers import SentenceTransformer
# 1. 加载CSV文件
df = pd.read_csv('book_info.csv')  # 替换为你的CSV文件路径


## Create item id lists and embeddings

item_ids = df['itemId'].tolist()
item_des = df['Book-Description'].tolist()

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
sentence_embeddings = model.encode(item_des,show_progress_bar=True)
norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)  # 计算每个向量的L2范数
normalized_embeddings = sentence_embeddings / norms  # 归一化
import numpy as np
np.save('item_embeddings.npy', normalized_embeddings)
np.save('item_ids.npy', item_ids)

item_id_to_embedding = {}
for i in range(len(item_ids)):
    item_id_to_embedding[item_ids[i]] = normalized_embeddings[i]

print(len(item_id_to_embedding.keys()))

import torch 
torch.save(item_id_to_embedding,'item_id_to_embedding.pt')


## Build User Database

from collections import defaultdict

def default_user_data():
    return {
        "passwd": "123456",
        "user_profile": [],
        "rated_items": []
    }
user_db = defaultdict(default_user_data)

import csv
from collections import defaultdict

def get_user_book_ratings(csv_file):
    user_books = defaultdict(list)
    
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 3:  # 确保至少有3列数据
                user = row[0].strip()
                book = row[1].strip()
                rating = row[2].strip()
                user_books[user].append((book, rating))
    
    return user_books


file_path = 'book_data.csv'  # 替换为你的文件路径
user_books = get_user_book_ratings(file_path)

def update_profile(user_db,user_name,item_id_to_embedding):
    user_rated = user_db[user_name]['rated_items']
    all_emb = 0
    all_w = 0
    for p_rate in user_rated:
        all_emb += item_id_to_embedding[p_rate[0]] * float(p_rate[1])
        all_w += float(p_rate[1])
    profile = all_emb/all_w
    user_db[user_name]['user_profile'] = profile


for user in user_books:
    user_db[user]['rated_items'] = user_books[user]
    update_profile(user_db,user,item_id_to_embedding)


torch.save(user_db,'user_db.pt')



