## ** User data
# user_id : XXXXXX
# delivery_tiem_flexibility : 0(flexible) ~ 10(as soon as possible)
# price_range : 0(don`t care) ~ 10(cheap)
# brand_loyalty : 0(don`t care) ~ 10(important)
# benefit_priority : 0(don`t care) ~ 10(important)
# product_quality : 0(don`t care) ~ 10(important)
## ** 

from dataclasses import dataclass
import string
from typing import List
from pprint import pprint
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



@dataclass
class Personality:
    delivery_tiem_flexibility : int
    price_range : int
    product_quality : int

@dataclass
class User:
    id : int
    age : int
    sex : string
    personality : Personality

@dataclass
class Users:
    users : List[User]

def train_model(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # 시퀀스 패딩
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

    # 입력과 레이블 분할
    input_sequences = input_sequences[:, :-1]
    labels = input_sequences[:, -1]

    # 모델 생성 (간단한 RNN 모델 사용)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(total_words, 64, input_length=max_sequence_len-1))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)))
    model.add(tf.keras.layers.Dense(total_words, activation='softmax'))

    # 모델 컴파일
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 모델 훈련
    model.fit(input_sequences, labels, epochs=30, verbose=1)

    return model, tokenizer, max_sequence_len

def get_trained_models(text_data):
    models = {}
    
    for type, data in text_data.items():
        model,tokenizer,max_sequence_len = train_model(data)

        models[f'{type}'] = [model,tokenizer,max_sequence_len]
    
    return models

def generate_keyword(model, tokenizer, text, max_sequence_len):
    # 키워드 num개 생성
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)

    predicted = np.array(predicted)

    # 확률 배열의 값을 확률로 간주하여 랜덤하게 인덱스 선택
    random_index = np.random.choice(len(predicted[0]), p=predicted[0])
    len_of_predicted = len(predicted[0]) - 1


    predicted_word = tokenizer.index_word[random_index%len_of_predicted + 1]
    text = predicted_word + " " + text

    return text

if __name__ == "__main__":
    file_path_user = 'user.csv'
    file_path_keyword = 'keyword.csv'

    # Load datas
    data_user = pd.read_csv(file_path_user)
    data_keyword = pd.read_csv(file_path_keyword)


    # Add users
    user_list = []
    for _, user in data_user.iterrows():
        personality=Personality(delivery_tiem_flexibility = user['delivery_tiem_flexibility'],
                                price_range = user['price_range'],
                                product_quality = user['product_quality'] )

        user_list.append(User(
            id=user['user_id'],
            age=user['age'], sex=user['sex'],
            personality=personality ))
        
    users = Users(user_list)

    # Train model by user`s personality
    models = get_trained_models(text_data=data_keyword)
    
    # Predict model by user`s personality
    for user in users.users:

        num_1st_personality = user.personality.delivery_tiem_flexibility
        num_2nd_personality = user.personality.product_quality
        num_3rd_personality = user.personality.price_range

        total = num_1st_personality + num_2nd_personality + num_3rd_personality
        ratio_of_1st_personality = num_1st_personality/total
        ratio_of_2nd_personality = num_2nd_personality/total
        ratio_of_3rd_personality = num_3rd_personality/total

        
        # 10개의 0~1의 난수 생성 
        random_numbers = [np.random.random() for _ in range(3)]

        input_text = "할랄고기"
        generated_keywords = []
        for random_num in random_numbers:
            if random_num <= ratio_of_1st_personality: model_types = 'delivery_tiem_flexibility'
            elif random_num <= ratio_of_1st_personality+ratio_of_2nd_personality: model_types = 'price_range'
            else : model_types = 'product_quality'

            model = models[model_types][0]
            tokenizer = models[model_types][1]
            max_sequence_len = models[model_types][2]

            generated_keywords.append(generate_keyword(model=model,
                                                       tokenizer=tokenizer,
                                                       text=input_text,
                                                       max_sequence_len=max_sequence_len))
        
        print(f"{user.id} 고객의 성향에 맞게 생성된 검색 키워드:", generated_keywords)
    print(data_user)