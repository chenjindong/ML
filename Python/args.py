
def demo1(*args):
    # args is a tuple
    print(type(args))
    print(args)


def demo2(**kwargs):
    # kwargs is a dict
    print(type(kwargs))
    print(kwargs)

demo1(1, 2, 3)
demo2(a=1, b=2, c=3)

