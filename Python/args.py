
def demo1(*args):
    print(type(args))
    print(args)


def demo2(**kwargs):
    print(type(kwargs))
    print(kwargs)

demo1(1, 2, 3)
demo2(a=1, b=2, c=3)

