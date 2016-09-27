class Student(object):
    '''
    fdasfda
    '''
    count = 0
    books = []

    def __init__(self, name, age):
        self.name = name
        self.age = age


print(Student.__name__)  # 特殊的类属性
# print(Student.__doc__)
print(Student.__bases__)  # 类的所有父类组成的元组
# print(Student.__dict__)
print(Student.__module__)  # 类所属的模块
print(Student.__class__)  # 类对象的类型


a = Student('cjd', 20)  # 建立对象
print(a.name)  # 访问类成员
