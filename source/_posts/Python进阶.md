---
title: Python进阶
categories:
  - Note
  - Python
abbrlink: 23063
date: 2023-11-23 17:13:24
---

##  魔术方法

* 非数学运算
    * 字符串表示
        * `__repr__`：测试语法调用的方法
        * `__str__`：print调用的方法
    * 集合、序列相关
        * `__len__`
        * `__getitem__`
        * `__setitem__`
        * `__delitem__`
        * `__contains__`
    * 迭代相关
        * `__iter__`
        * `__next__`
    * 可调用
        * `__call__`
    * with 上下文管理器
        * `__enter__`
        * `__exit__`
    * 数值转换
        * `__abs__`
        * `__bool__`
        * `__int__`
        * `__float__`
        * `__hash__`
        * `__index__`
    * 元类相关
        * `__new__`
        * `__init__`
    * 属性相关
        * `__getattr__`、`__setattr__`
        * `__getattribute__`、`__setattribute__`
        * `__dir__`
    * 属性描述符
        * `__get__`、`__set__`、`__delete__`
    * 协程
        * `__await__`、`__aiter_`、`__anext__`、`__aenter__`、`__aexit__`
* 数学运算
    * 一元运算符
        * `__neg__` （-）、`__pos__`（+）、`__abs__`
    * 二元运算符
        * `__lt__`（$\lt$）、`__le__`（$\le$）、`__eq__`（$==$）、`__gt__`（$\gt$）、`__ge__`（$\ge$）
    * 算数运算符
        * `__add__` （+）、`__sub__` （-）、`__mul__` （*）、`__truediv__` （/）、`__floordiv__` （//）、`__mod__` （%）、`__divmod__` （divmod()）、`__pow__` （\*\*或pow()）、`__round__` （round()）
    * 反向算数运算符
        * `__radd__`、`__rsub__`、`__rmul__`、`__rtruediv__`、`__rfloordiv__`、`__rmod__`、`__rdivmod__`、`__rpow__`
    * 增量赋值算数运算符
        * `__iadd__`、`__isub__`、`__imul__`、`__itruediv__`、`__ifloordiv__`、`__imod__`、`__ipow__`
    * 位运算符
        * `__invert__` （~）、`__lshift__` （<<）、`__rshift__` （>>）、`__and__` （&）、`__or__` （|）、`__xor__` （^）
    * 反向运算符
        * `__rlshift__` 、`__rrshift__`、`__rand__`、`__ror__` 、`__rxor__` 
    * 增量赋值位运算符
        * `__ilshift__` 、`__irshift__`、`__iand__`、`__ior__` 、`__ixor__` 

## 类和对象

```python
class Student:
    
    age: int = 10
        
    def __init__(self, name, age):
        self.__name = name  # 私有属性
        self.age = age
    
    # 实例方法：第一个参数是self，可以访问类中的所有属性和方法
    def func1(self):
        pass
    
    # 静态方法：无法访问类中的属性
    @staticmethod
    def func2():
        print("staticmethod")
        
    # 类方法：可以被类对象和示例对象调用，类方法可以访问类属性
    @classmethod
    def func3(cls):
        return cls.age
```

* 自省机制

    * python中对象都具有一个特殊的属性：`__dict__` 只能查询属于自身的属性

        ```python
        stu = Student('zoe', 21)
        print(stu.__dict__)
        ```

    * `dir`函数，可以查询一个对象中所有的属性和方法，包含这个对象的父类

        ```python
        print(dir(stu))
        ```

* `contextlib`完成上下文管理器

    ```python
    import contextlib
    
    
    @contextlib.contextmanager
    def open_file(file_name):
        print(f'open: {file_name}')
        # 被 contextmanager 装饰的函数必须是一个生成器函数
        # enter 的代码要在 yield 之上
        # exit 的代码要在 yield 之x
        yield {'name': 'zoe', 'age': 21}
        print(f'close: {file_name}')
    
        
    with open_file('zoe.txt') as f_open:
        print("Run...")
        
        
    """output
    open: zoe.txt
    Run...
    close: zoe.txt
    """
    ```

    

    
