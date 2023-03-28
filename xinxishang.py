import re
import os
import math
import jieba
from collections import Counter

# 读取小说文本数据
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    return content

# 文本预处理
def preprocess(text):
    content = re.sub(r'[^\u4e00-\u9fa5]+', '', text)
    return content

# 分词
def tokenize(text):
    words = list(jieba.cut(text))
    return words

# 去除停用词
def remove_stopwords(words):
    with open('stopwords.txt', 'r', encoding='utf-8', errors='ignore') as f:
        stopwords = set(line.strip() for line in f.readlines())
    filtered_words = [word for word in words if word not in stopwords]
    return filtered_words

# 计算N元信息熵
def compute_ngram_entropy(elements, n):
    ngrams = [tuple(elements[i:i+n]) for i in range(len(elements)-n+1)]
    counter = Counter(ngrams)
    total_count = sum(counter.values())
    entropy = 0
    for count in counter.values():
        p = count / total_count
        entropy -= p * math.log2(p)
    return entropy

# 主程序
def main():
    novels = ['Jinyong/白马啸西风.txt',
              'Jinyong/碧血剑.txt',
              'Jinyong/飞狐外传.txt',
              'Jinyong/连城诀.txt',
              'Jinyong/鹿鼎记.txt',
              'Jinyong/三十三剑客图.txt',
              'Jinyong/射雕英雄传.txt',
              'Jinyong/神雕侠侣.txt',
              'Jinyong/书剑恩仇录.txt',
              'Jinyong/天龙八部.txt',
              'Jinyong/侠客行.txt',
              'Jinyong/笑傲江湖.txt',
              'Jinyong/雪山飞狐.txt',
              'Jinyong/倚天屠龙记.txt',
              'Jinyong/鸳鸯刀.txt',
              'Jinyong/越女剑.txt'
              ]  # 列出所有小说的文件名
    results = []

    for novel in novels:
        text = read_file(novel)
        cleaned_text = preprocess(text)
        tokens = tokenize(cleaned_text)
        words = remove_stopwords(tokens)

        char_unigram_entropy = compute_ngram_entropy(cleaned_text, 1)
        char_bigram_entropy = compute_ngram_entropy(cleaned_text, 2)
        char_trigram_entropy = compute_ngram_entropy(cleaned_text, 3)

        word_unigram_entropy = compute_ngram_entropy(words, 1)
        word_bigram_entropy = compute_ngram_entropy(words, 2)
        word_trigram_entropy = compute_ngram_entropy(words, 3)

        results.append((novel, char_unigram_entropy, char_bigram_entropy, char_trigram_entropy,
                        word_unigram_entropy, word_bigram_entropy, word_trigram_entropy))

    # 输出结果
    print("小说 | 字一元熵 | 字二元熵 | 字三元熵 | 词一元熵 | 词二元熵 | 词三元熵")
    print("---------------------------------------------------------------")
    for novel, char_unigram_entropy, char_bigram_entropy, char_trigram_entropy, \
        word_unigram_entropy, word_bigram_entropy, word_trigram_entropy in results:
        print(f'{novel}: {char_unigram_entropy:.4f}, {char_bigram_entropy:.4f}, {char_trigram_entropy:.4f}, '
              f'{word_unigram_entropy:.4f}, {word_bigram_entropy:.4f}, {word_trigram_entropy:.4f}')

if __name__ == '__main__':
    main()
