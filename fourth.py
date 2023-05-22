import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout

# 金庸的16部小说存放在'novels'文件夹中
file_list = os.listdir('novels')


raw_text = ''
for file_name in file_list:
    with open(os.path.join('novels', file_name), 'r', encoding='utf-8', errors='ignore') as file:
        raw_text += file.read()

# 使用每个字符作为一个词
tokens = list(raw_text)

# 创建char-to-integer映射和integer-to-char映射
char_to_int = {char: i for i, char in enumerate(set(tokens))}
int_to_char = {i: char for i, char in enumerate(set(tokens))}

sequence_length = 20  # 将序列长度减小以节省内存
def generate_sequences(tokens, char_to_int, sequence_length, batch_size):
    while True:
        for i in range(0, len(tokens) - sequence_length, batch_size):
            inputs, outputs = [], []
            for j in range(min(batch_size, len(tokens) - sequence_length - i)):
                inputs.append([char_to_int[char] for char in tokens[i+j:i+j + sequence_length]])
                outputs.append(char_to_int[tokens[i+j + sequence_length]])
            inputs = np.array(inputs)
            outputs = to_categorical(outputs, num_classes=len(set(tokens)))
            yield inputs, outputs

model = Sequential()
model.add(Embedding(len(set(tokens)), 50, input_length=sequence_length))  # Embedding 层
model.add(LSTM(128))  # LSTM层
model.add(Dense(len(set(tokens)), activation='softmax'))  # 输出层

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam')

# 创建生成器
batch_size = 256
sequence_generator = generate_sequences(tokens, char_to_int, sequence_length, batch_size)

# 训练模型
model.fit(sequence_generator, steps_per_epoch=len(tokens)//(sequence_length*batch_size), epochs=50)
def generate_text(seed_text, model, int_to_char, char_to_int, sequence_length):
    seed_text = list(seed_text)
    seed_text = [char_to_int.get(char, 0) for char in seed_text]
    seed_text = pad_sequences([seed_text], maxlen=sequence_length, truncating='pre')

    generated_text = ''

    for _ in range(100):  # 生成100个字符
        prediction = model.predict(seed_text)
        prediction = np.argmax(prediction, axis=-1)
        generated_char = int_to_char[prediction[0]]
        generated_text += generated_char

        seed_text = np.concatenate((seed_text[:, 1:], prediction.reshape(1, 1)), axis=1)

    return generated_text

seed_text = "种子文本"
while len(seed_text) < sequence_length:  # 确保种子文本长度和序列长度相等
    seed_text = ' ' + seed_text

generated_text = generate_text(seed_text, model, int_to_char, char_to_int, sequence_length)
print(generated_text)
