# life is short, you need study python everyday.
# author    TuringEmmy
# time      2018/11/20 15:11
# project   MachineLearning


def a(func):
    def a1(a, b):
        n = func(a, b)
        return n + 3

    return a1


@a
def b(x, y):
    return x + y


print(b(1, 2))
