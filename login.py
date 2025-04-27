from flask import Flask, request, jsonify
from flask_cors import CORS
import torch 
import numpy as np 
from algo import return_books_random,tranform_selected_to_real_rating,updated_user_item_rating,update_userdb_and_profile,hybrid_recommendation
app = Flask(__name__)
CORS(app)

# MySQL 配置
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'book_recommender'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'


user_db_path = 'user_db.pt'
user_rating_path = 'book_data.csv'

user_db = torch.load('user_db.pt')
item_lists = np.load('item_ids.npy')
item_embeddings = np.load('item_embeddings.npy')
item_id_to_embedding = torch.load('item_id_to_embedding.pt')



@app.route('/api/auth/register', methods=['POST'])
def register(): ## Register new user and return book ids for new user to chose preference
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password are required'}), 400

    try:

        # 检查用户名是否已存在
        
        if username in user_db:
            return jsonify({'success': False, 'message': 'Username already exists'}), 400
        else:
        # 创建新用户，默认 has_set_preferences 为 False
            user_db[username]['passwd'] = password
            book_ids_for_chosen = return_books_random(item_lists,k=30)
            return jsonify({
                'success': True,
                'message': 'User registered successfully',
                'book_ids_for_chosen':book_ids_for_chosen,
                'user': {
                    'username': username,
                    'has_set_preferences': False
                }
            }), 201
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password are required'}), 400

    try:
        
        if (username in user_db)&(user_db[username]['passwd']==password):
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'user': {
                    'username': user_db[username]
                    'has_set_preferences': bool(user_db[username]['rated_items'])
                }
            })
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500



@app.route('/api/set_preferences', methods=['POST'])
def set_preferences():
    """设置初始偏好"""
    data = request.get_json()
    username = data.get('username')
    selected = data.get('selected_books', [])  # 用户选择的书籍ID列表 [('item_id1','like'),('item_id2','like'),('item_id3','like')]
    
    if username not in users_db:
        return jsonify({'error': 'user does not exist'}), 404
        
    if len(selected) < 3:
        return jsonify({'error': 'please select at least 3 books'}), 400
    
    selected = tranform_selected_to_real_rating(selected)

    updated_user_item_rating(username,selected,user_rating_path)
    update_userdb(username,selected,user_db_path)
    

    return jsonify({
        'success': True,
        'message': 'Updated new registered user preference and profile'
    })

@app.route('/api/gen_recommendation', methods=['POST'])
def generate_recommendations(username):
    """核心推荐逻辑"""
    data = request.get_json()
    username = data.get('username')
    

    hybrid_recommendation_list = hybrid_recommendation(username,user_rating_path,user_db,item_lists,item_embeddings,k=10)
    
    return jsonify({
        'recommendation_list': hybrid_recommendation_list,
        'message': 'have returned recommendation'
    })

@app.route('/api/feedback', methods=['POST'])
def handle_feedback(username):
    data = request.get_json()
    username = data.get('username')
    selected = data.get('feedback', []) # 用户选择的书籍ID列表 [('item_id1','like'),('item_id2','dislike'),('item_id3','like')]
    
    if username not in users_db:
        return jsonify({'error': 'user does not exist'}), 404

    selected = tranform_selected_to_real_rating(selected)

    updated_user_item_rating(username,selected,user_rating_path)
    update_userdb_and_profile(username,selected,user_db_path,item_id_to_embedding)
    

    return jsonify({
        'success': True,
        'message': 'already handle the feedback and update the profile'
    })




if __name__ == '__main__':
    app.run(debug=True, port=5000)

