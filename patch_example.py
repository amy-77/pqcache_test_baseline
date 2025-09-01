# 补丁概念演示

# 1. 原始的"人"类
class Person:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return f"{self.name}: 你好"

# 2. 创建一个人
xiaoming = Person("小明")
print("补丁前:")
print(xiaoming.speak())  # 输出: 小明: 你好

# 3. 定义补丁函数
def add_english_patch(person):
    """给Person添加说英语的能力"""
    
    # 保存原始的speak方法
    original_speak = person.speak
    
    # 定义新的speak方法
    def new_speak(self):
        chinese = original_speak()  # 调用原始功能
        english = f"{self.name}: Hello"  # 添加新功能
        return f"{chinese} | {english}"
    
    # "偷偷替换"原始方法
    import types
    person.speak = types.MethodType(new_speak, person)
    
    return person

# 4. 给小明打补丁
xiaoming = add_english_patch(xiaoming)

print("\n补丁后:")
print(xiaoming.speak())  # 输出: 小明: 你好 | 小明: Hello

# 5. 补丁的好处
print("\n补丁的好处:")
print("✅ 小明还是原来的小明（Person类）")
print("✅ 但现在他会说英语了")
print("✅ 我们没有修改Person类的源代码")
print("✅ 其他Person实例不受影响")

# 验证其他Person不受影响
xiaohong = Person("小红")
print(f"小红(未打补丁): {xiaohong.speak()}") 