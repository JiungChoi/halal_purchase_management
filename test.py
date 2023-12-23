import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 데이터프레임 생성 (고객의 성향)
data = {
    'delivery_time_flexibility': [9, 2, 5, 6, 7],
    'price_range': [2, 6, 5, 2, 1],
    'brand_loyalty': [2, 3, 4, 4, 5],
    'benefit_priority': [5, 5, 5, 5, 5],
    'product_quality': [8, 7, 6, 3, 1]
}
df = pd.DataFrame(data)

# 텍스트 데이터 (고객의 성향에 맞는 키워드)
text_data = [
    "할랄 고기 배송이 빠른 할랄 고기 할랄 한우",
    "빠른 배송 신선한 육류 한우 등심",
    "신선한 과일 배송이 빠른 과일 시리얼",
    # 추가 데이터셋
]

# 토큰화
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
total_words = len(tokenizer.word_index) + 1

# 시퀀스 생성
input_sequences = []
for line in text_data:
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
model.fit(input_sequences, labels, epochs=100, verbose=1)

# 고객의 성향을 입력으로 받아 해당 성향에 맞는 검색 키워드 생성 (예측)
def generate_keywords(model, tokenizer, text, max_sequence_len):
    for _ in range(10):  # 새로운 키워드를 10개 생성
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word = tokenizer.index_word[predicted.argmax(axis=-1)[0]]
        text += " " + predicted_word
    return text

# 고객의 성향 데이터를 가지고 키워드 생성
customer_tendencies = "delivery_time_flexibility price_range brand_loyalty benefit_priority product_quality"
resulting_keywords = generate_keywords(model, tokenizer, customer_tendencies, max_sequence_len)
print("고객의 성향에 맞게 생성된 검색 키워드:", resulting_keywords)
