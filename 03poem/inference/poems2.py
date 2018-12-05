# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/5/18 10:27 PM
# project   DeepLearingStudy
import collections

from inference.poems import start_token, end_token


def process_poems(file_name):
    poems = []

    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            title, content = line.strip().split(':')

            # 也就是所说的数据清洗
            # 对于诗的乱符
            if '_' in content or '[' in content:
                continue
            if len(content) < 5 or len(content) > 80:
                continue
            # 开始自定义的字符加上自己的内容
            content = start_token + content + end_token

            poems.append(content)

    poems = sorted(poems, key=lambda l: len(line))

    all_words = []
    # 有一个字，就加一个
    for poem in poems:
        all_words += [word for word in poem]

    # 统计词频
    counter = collections.Counter(all_words)
    # -1即代表出现的个数
    count_paris = sorted(counter.items(), key=lambda x: x[-1])
    # 拿出单独的词
    words, = zip(*count_paris)
    words = words[:len(words)]
    # 字转换成int
    # 这是一个映射表
    word_int_map = dict(zip(words, range(len(words))))

    # 转换成向量
    poems_vector = [list(map(lambda word: word_int_map.get(word, len(words))))]

    return poems_vector, word_int_map, words
