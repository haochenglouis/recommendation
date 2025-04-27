import pandas as pd
import torch
import os
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from surprise import Dataset
from surprise import Reader
from surprise import SVD
import random 


def return_books_random(all_books_id, k):
    all_books_id = list(all_books_id)
    return random.sample(all_books_id, k)

def tranform_selected_to_real_rating(selected):
    real_rating = []
    for sel in selected:
        if sel[1] = 'like':
            real_rating.append((sel[0],'9'))
        elif sel[1] = 'dislike':
            real_rating.append((sel[0],'3'))
    return real_rating

def updated_user_item_rating(username,selected,user_rating_path):
    df = pd.read_csv(user_rating_path, header=None)
    for sel in selected:
        df = pd.concat([df, pd.DataFrame([username,sel[0],sel[1]])], ignore_index=True)
    df.to_csv(user_rating_path, header=False, index=False)

def update_userdb_and_profile(username,selected,user_db_path,item_id_to_embedding):
    current_userdb = torch.load(user_db_path)
    current_userdb[username]['rated_items'] = current_userdb[username]['rated_items'] + selected
    user_rated = current_userdb[user_name]['rated_items']
    all_emb = 0
    all_w = 0
    for p_rate in user_rated:
        all_emb += item_id_to_embedding[p_rate[0]] * float(p_rate[1])
        all_w += float(p_rate[1])
    profile = all_emb/all_w
    current_userdb[user_name]['user_profile'] = profile
    torch.save(current_userdb,user_db_path)




def weighted_borda_count(rank1, rank2, weight1, weight2, k):
    """
    使用加权 Borda Count 方法融合两个推荐算法的排名结果
    
    参数:
    - rank1: 算法1的推荐列表 (top k)
    - rank2: 算法2的推荐列表 (top k)
    - weight1: 算法1的权重 (如 0.3)
    - weight2: 算法2的权重 (如 0.7)
    - k: 需要返回的最终推荐数量
    
    返回:
    - 最终推荐列表 (按加权 Borda 总分排序)
    """
    borda_scores = defaultdict(float)
    
    # 计算算法1的加权 Borda 得分
    for idx, item in enumerate(rank1):
        borda_scores[item] += (len(rank1) - idx) * weight1
    
    # 计算算法2的加权 Borda 得分
    for idx, item in enumerate(rank2):
        borda_scores[item] += (len(rank2) - idx) * weight2
    
    # 按加权 Borda 总分降序排序，同分时优先选择在 rank1 中排名更高的
    sorted_items = sorted(
        borda_scores.items(),
        key=lambda x: (-x[1], rank1.index(x[0]) if x[0] in rank1 else float('inf'))
    )
    
    # 提取前 k 个 item
    final_ranking = [item for item, score in sorted_items[:k]]
    
    return final_ranking



def train_svd(user_rating_path):
    reader = Reader(line_format='user item rating', sep=',',rating_scale=(1, 10)  )
    data = Dataset.load_from_file(user_rating_path, reader=reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    return algo

def svd_recommend(algo,username,book_ids,k):
    prediction = []
    for book_id in book_ids:
        prediction.append(algo.predict(username, book_id).est)
    zipped = sorted(zip(prediction, book_ids), reverse=True)
    sorted_pred, sorted_book_ids = zip(*zipped)
    return sorted_book_ids[:k]


def content_based_recommend(user_profile,item_lists,item_embeddings,user_unrated_book_ids,k):
    user_profile = np.array(user_profile).reshape(1, -1)
    similarities = cosine_similarity(user_profile, item_embeddings).flatten()
    item_similarity_pairs = list(zip(item_lists, similarities))
    unrated_items = [
        (item_id, sim) 
        for item_id, sim in item_similarity_pairs 
        if item_id in user_unrated_book_ids
    ]
    unrated_items_sorted = sorted(unrated_items, key=lambda x: x[1], reverse=True)
    recommended_items = [item[0] for item in unrated_items_sorted]
    return recommended_items[:k]



def hybird_recommendation(username,user_rating_path,user_db,item_lists,item_embeddings,k=10):
	user_profile = user_db[user_name]['rated_items']['user_profile']
    user_rated = user_db[user_name]['rated_items']
    user_rated_book_ids = [i[0] for i in user_rated]
    user_unrated_book_ids = list(set(item_lists) - set(user_rated_book_ids))
    algo = train_svd(user_rating_path)
    svd_recommend_book_ids = svd_recommend(algo,username,user_unrated_book_ids,k)
    content_recommend_book_ids = content_based_recommend(user_profile,item_lists,item_embeddings,user_unrated_book_ids,k)
    final_hybrid_ranking = weighted_borda_count(svd_recommend_book_ids, content_recommend_book_ids,0.7,0.3,k)
    return final_hybrid_ranking


