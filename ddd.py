import os
import re
import numpy as np
import pandas as pd
import jieba
from gensim import corpora, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt


# 数据预处理
def preprocess(text, unit='word'):
    stopwords = set(line.strip() for line in open('stopwords.txt', encoding='utf-8',errors='ignore'))
    if unit == 'word':
        words = jieba.cut(text)
        words = [word for word in words if word not in stopwords and len(word) > 1]
    else:
        words = [c for c in text if c not in stopwords]
    return words


# 从文本中提取200个段落
def extract_paragraphs(text):
    paragraphs = re.split('[\n\r]', text)
    paragraphs = [p.strip() for p in paragraphs if len(p) > 500]
    return paragraphs[:200]


# 读取数据并进行预处理
def load_data(unit='word'):
    data = []
    for novel in os.listdir('novels'):
        with open(os.path.join('novels', novel), 'r', encoding='utf-8',errors='ignore') as f:
            text = f.read()
            paragraphs = extract_paragraphs(text)
            for paragraph in paragraphs:
                words = preprocess(paragraph, unit)
                data.append({'text': words, 'label': novel[:-4]})
    return data


# LDA模型构建
def train_lda(corpus, dictionary, num_topics):
    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state=42)
    return lda


# 分类器训练
def train_classifier(X_train, y_train):
    classifier = SVC()
    classifier.fit(X_train, y_train)
    return classifier


# 分类器评估
def evaluate_classifier(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# 加载数据
data_word = load_data(unit='word')
data_char = load_data(unit='char')

topic_numbers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# 实验函数
def run_experiment(data, unit):
    df = pd.DataFrame(data)
    texts = df['text'].tolist()
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    topic_numbers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    accuracies = []

    for num_topics in topic_numbers:
        lda = train_lda(corpus, dictionary, num_topics)
        topic_distributions = []
        for i in range(len(corpus)):
            topic_probs = lda.get_document_topics(corpus[i], minimum_probability=0)
            topic_probs = [prob for _, prob in sorted(topic_probs, key=lambda x: x[0])]
            topic_distributions.append(topic_probs)
        topic_distributions = np.array(topic_distributions)
        X = topic_distributions
        y = df['label'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        classifier = train_classifier(X_train, y_train)
        accuracy = evaluate_classifier(classifier, X_test, y_test)
        accuracies.append(accuracy)

    return accuracies

# 运行实验
word_accuracies = run_experiment(data_word, 'word')
char_accuracies = run_experiment(data_char, 'char')

# 结果可视化
plt.plot(topic_numbers, word_accuracies, marker='o', label='Word-based')
plt.plot(topic_numbers, char_accuracies, marker='s', label='Char-based')
plt.xlabel('Number of Topics')
plt.ylabel('Accuracy')
plt.title('Classification Performance of LDA with Different Number of Topics')
plt.legend()
plt.grid()
plt.show()

# 输出实验结果
result_table = pd.DataFrame({'Number of Topics': topic_numbers,
                             'Word-based Accuracy': word_accuracies,
                             'Char-based Accuracy': char_accuracies})

# 在控制台显示表格
print(result_table)
