class Person:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.age = kwargs.get('age')

    # 类方法
    @classmethod
    def method(cls):
        print('this is class method')

    # 属性函数
    @property
    def get_info(self):
        return self.name + '\t' + str(self.age)

if __name__ == '__main__':
    # test 类方法
    Person.method()

    # test 属性函数
    p = Person(name='cjd', age=20)
    print(p.get_info)