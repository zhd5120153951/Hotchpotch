'''
@FileName   :__init__函数通俗解释.py
@Description:
@Date       :2020s/09/28 11:29:57
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''


class People(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age
        return

    """
    总结：当使用print输出对象的时候，只要自己定义了__str__(self)方法，那么就会打印从在这个方法中return的数据。__str__方法需要返回一个字符串，当做这个对象的描写。
    """

    def __str__(self):
        return self.name + ":" + str(self.age)

    def __lt__(self, other):
        return self.name < other.name if self.name != other.name else self.age < other.age


# 常见的魔法函数
class FirstMagicMothod:
    # 类属性和类方法  不同于 实例属性和实例方法
    # 类的构造
    def __init__(self):
        print("调用构造方法")

    # 类的属性
    url = "https://github.com/zhd5120153951"

    # 实例方法
    def speak(self, content):
        print(content)


class CLanguage:
    def __init__(self, name, add):
        print(name, "sdf", add)
        self.name = name
        self.add = add

    """
    __repr__() 方法是类的实例化对象用来做“自我介绍”的方法，默认情况下，它会返回当前对象的“类名+object at+内存地址”，而如果对该方法进行重写，可以为其制作自定义的自我描述信息。
    """

    def __repr__(self) -> str:
        return "CLanguage = " + self.name + self.add


###未完...

if __name__ == "__main__":
    p_list = [People("abc", 18), People("abd", 20), People("abe", 29)]
    print("\t".join([str(item) for item in sorted(p_list)]))

    ret = FirstMagicMothod()
    print(FirstMagicMothod.url)
    # FirstMagicMothod.speak()错误用法

    print(ret.url)
    ret.speak("实例方法")

    CL = CLanguage("adad", "dwrr")
    print(CL)