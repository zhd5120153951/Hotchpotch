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


###未完...

if __name__ == "__main__":
    p_list = [People("abc", 18), People("abd", 20), People("abe", 29)]
    print("\t".join([str(item) for item in sorted(p_list)]))

    ret = FirstMagicMothod()
    print(FirstMagicMothod.url)
    # FirstMagicMothod.speak()错误用法

    print(ret.url)
    ret.speak("实例方法")

    add = CLanguage("adad", "dwrr")
