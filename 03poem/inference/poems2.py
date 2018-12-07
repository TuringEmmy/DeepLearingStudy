# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/5/18 10:27 PM
# project   DeepLearingStudy
import collections

from inference.poems import start_token, end_token


# 预处理
def process_poems(file_name):
    # 处理好的结果整成list
    poems = []

    with open(file_name, 'r', encoding='utf-8') as f:
        # 按行进行读取
        for line in f.readlines():
            # 进行分割（题目,内容）
            title, content = line.strip().split(':')

            # =========也就是所说的数据清洗======

            # 对于诗的乱符
            if '_' in content or '[' in content:
                continue

            # 进行长度的判断
            if len(content) < 5 or len(content) > 80:
                continue

            # 开始自定义的字符加上自己的内容
            content = start_token + content + end_token

            poems.append(content)

    # 把list进行排序，句子越长，排在前面
    poems = sorted(poems, key=lambda l: len(line))

    # 统计每个词出现的次数 
    all_words = []

    # 有一个字，就加一个
    for poem in poems:
        all_words += [word for word in poem]

    # 统计词频（自动统计）  如果有偏僻字，直接过滤
    counter = collections.Counter(all_words)

    # -1即代表出现的个数
    count_paris = sorted(counter.items(), key=lambda x: x[-1])

    # 拿出单独的词
    words, _ = zip(*count_paris)
    # 取所有的
    words = words[:len(words)]

    # 字转换成int
    # 这是一个映射表
    word_int_map = dict(zip(words, range(len(words))))

    # 转换成向量，在进行映射
    poems_vector = [list(map(lambda word: word_int_map.get(word, len(words))))]

    # 这个Vector相当于一个字
    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vector, word_to_int):
    n_chunk = len(poems_vector) / batch_size
    # 填充数据
    x_batches = []
    y_batches = []

    # 跳转进行
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        batches = poems_vector[start_index:end_index]

        # 按照最长的诗句进行
        length = max(map(len, batches))
        # 填充空格
        x_data = np.full(batch_size, length, word_to_int(' '), np.int32)

        for row in range(batch_size):
            x_data[row, :len(batch_sizes[row])] = batches[row]

        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]

        x_batch
