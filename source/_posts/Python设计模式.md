---
title: Python设计模式
categories:
  - Note
  - Python
abbrlink: 29319
date: 2023-11-23 16:22:44
---

## 0 设计模式

对软件中普遍存在（反复出现）的各种问题，所提出的解决方案。每一个设计模式系统地命名、解释和评价了面向对象系统中一个重要的和反复出现的设计。

* 面向对象设计的SOLID原则

    * 开放封闭原则：一个软件实体，如类、模块和函数应该对扩展开放，对修改关闭。即，软件实体应尽量在不修改原有代码的情况下进行扩展

    * 里氏替换原则：引用所有父类的地方必须能透明地使用其子类的对象。

    * 依赖倒置原则

        要针对接口编程，而不是针对实现编程

        * 高层模块不应该依赖底层模块，二者都应该依赖其抽象；
        * 抽象不应该依赖细节；
        * 细节应该依赖抽象。

    * 接口隔离原则：使用多个专门的接口，而不使用单一的总接口，即客户端不应该依赖那些它不需要的接口

    * 单一职责原则：不要存在多余一个导致类变更的原因。通俗地说，即一个类只负责一项职责

* 设计模式分类

    * 创建型模式（5种）：工厂方法模式、抽象工厂模式、创建者模式、原型模式、单例模式
    * 结构型模式（7种）：适配器模式、桥模式、组合模式、装饰模式、外观模式、享元模式、代理模式
    * 行为型模式（11种）：解释器模式、责任链模式、命令模式、迭代器模式、中介者模式、备忘录模式、观察者模式、状态模式、策略模式、访问者模式、模板方法模式

## 1 创建型模式

### 1.1 简单工厂模式

* 内容：不直接向客户端暴露对象创建的实现细节，而是通过一个工厂类来负责创建产品类的实例。
* 角色
    * 工厂角色（Creator）
    * 抽象产品角色（Product）
    * 具体产品角色（Concrete Product）
* 优点
    * 隐藏了对象创建的实现细节
    * 客户端不需要修改代码
* 缺点
    * 违反了单一职责原则，将创建逻辑集中到一个工厂类里
    * 当添加新产品时，需要修改工厂代码，违反了开闭原则

```python
from abc import ABCMeta, abstractmethod


class Payment(metaclass=ABCMeta):
    # 抽象产品
    @abstractmethod
    def pay(self, money):
        pass


class Alipay(Payment):
    # 具体产品
    def pay(self, money):
        print("支付宝支付%d原." % money)


class WechatPay(Payment):
    # 具体产品
    def pay(self, money):
        print("微信支付%d元" % money)


class PaymentFactory:
    # 工厂
    def create_payment(self, method):
        if method == 'alipay':
            return Alipay()
        elif method == 'wechat':
            return WechatPay()
        else:
            raise TypeError("No such payment named %s" % method)


# client
pf = PaymentFactory()
p = pf.create_payment('alipay')
p.pay(100)

```

### 1.2 工厂方法模式

* 内容：定义一个用于创建对象的接口（工厂接口），让子类决定实例化哪一个产品类
* 角色
    * 抽象工厂角色（Creator）
    * 具体工厂角色（Concrete Creator）
    * 抽象产品角色（Product）
    * 具体产品角色（Concrete Product）
* 优点
    * 每个具体产品都对应一个具体工厂类，不需要修改工厂类代码
    * 隐藏了对象创建的实现细节
* 缺点
    * 每增加一个具体产品类，就必须增加一个相应的具体工厂类

```python
from abc import ABCMeta, abstractmethod


class Payment(metaclass=ABCMeta):
    # 抽象产品
    @abstractmethod
    def pay(self, money):
        pass


class Alipay(Payment):
    # 具体产品
    def __init__(self, use_huabei=False):
        self.use_huabei = use_huabei

    def pay(self, money):
        if self.use_huabei:
            print("花呗支付%d原." % money)
        else:
            print("支付宝支付%d原." % money)


class WechatPay(Payment):
    # 具体产品
    def pay(self, money):
        print("微信支付%d元" % money)


class PaymentFactory(metaclass=ABCMeta):
    # 抽象工厂
    @abstractmethod
    def create_payment(self):
        pass


class AlipayFactory(PaymentFactory):
    # 具体工厂
    def create_payment(self):
        return Alipay()


class WechatFactory(PaymentFactory):
    # 具体工厂
    def create_payment(self):
        return WechatPay()


class HuabeiFactory(PaymentFactory):
    # 具体工厂
    def create_payment(self):
        return Alipay(use_huabei=True)


# client
pf = HuabeiFactory()
p = pf.create_payment()
p.pay(100)
```

### 1.3 抽象工厂模式

* 内容：定义一个工厂类接口，让工厂子类来创建一系列相关或相互依赖的对象

    相比工厂方法模式，抽象工厂模式种的每个具体工厂都生产一套产品

* 角色

    * 抽象工厂角色（Creator）
    * 具体工厂角色（Concrete Creator）
    * 抽象产品角色（Product）
    * 具体产品角色（Concrete Product）
    * 客户端（Client）

* 优点

    * 将客户端与类的具体实现相分离
    * 每个工厂创建了一个完整的产品系列，使得易于交换产品系列
    * 有利于产品的一致性（即产品之间的约束关系）

* 缺点

    * 难以支持新种类的（抽象）产品

```python
from abc import ABCMeta, abstractmethod


# ------抽象产品------

class PhoneShell(metaclass=ABCMeta):
    @abstractmethod
    def show_shell(self):
        pass


class CPU(metaclass=ABCMeta):
    @abstractmethod
    def show_cpu(self):
        pass


class OS(metaclass=ABCMeta):
    @abstractmethod
    def show_os(self):
        pass


# ------抽象工厂------

class PhoneFactory(metaclass=ABCMeta):
    @abstractmethod
    def make_shell(self):
        pass

    @abstractmethod
    def make_cpu(self):
        pass

    @abstractmethod
    def make_os(self):
        pass


# ------具体产品------

class SmallShell(PhoneShell):
    def show_shell(self):
        print("普通手机小手机壳")


class BigShell(PhoneShell):
    def show_shell(self):
        print("普通手机大手机壳")


class AppleShell(PhoneShell):
    def show_shell(self):
        print("苹果手机壳")


class SnapDragonCPU(CPU):
    def show_cpu(self):
        print("骁龙CPU")


class MediaTekCPU(CPU):
    def show_cpu(self):
        print("联发科CPU")


class AppleCPU(CPU):
    def show_cpu(self):
        print("苹果CPU")


class Android(OS):
    def show_os(self):
        print("Android系统")


class IOS(OS):
    def show_os(self):
        print("iOS系统")


# ------具体工厂------

class MiFactory(PhoneFactory):
    def make_cpu(self):
        return SnapDragonCPU()

    def make_os(self):
        return Android()

    def make_shell(self):
        return BigShell()


class HuaweiFactory(PhoneFactory):
    def make_cpu(self):
        return MediaTekCPU()

    def make_os(self):
        return Android()

    def make_shell(self):
        return SmallShell()


class IPhoneFactroy(PhoneFactory):
    def make_cpu(self):
        return AppleCPU()

    def make_os(self):
        return IOS()

    def make_shell(self):
        return AppleShell()


# ------客户端------

class Phone:
    def __init__(self, cpu, os, shell):
        self.cpu = cpu
        self.os = os
        self.shell = shell

    def show_info(self):
        print("手机信息：")
        self.cpu.show_cpu()
        self.os.show_os()
        self.shell.show_shell()


def make_phone(factory):
    cpu = factory.make_cpu()
    os = factory.make_os()
    shell = factory.make_shell()
    return Phone(cpu, os, shell)


p1 = make_phone(MiFactory())
p1.show_info()
```

### 1.4 建造者模式

* 内容：将一个复杂对象的构建与它的表示分离，使得同样的构建过程可以创建不同的表示

    建造者模式与抽象工厂模式相似，也用来创建复杂对象。主要区别是建造者模式着重步步构造一个复杂对象，而抽象工厂模式着重于多个系列的产品对象。

* 角色

    * 抽象建造者（Builder）
    * 具体建造者（Concrete Builder）
    * 指挥者（Director）
    * 产品（Product）

* 优点

    * 隐藏了一个产品的内部结构和装配过程
    * 将构造代码与表示代码分开
    * 可以对构造过程进行更精细的控制

```python
from abc import ABCMeta, abstractmethod


class Player:
    def __init__(self, face=None, body=None, arm=None, leg=None):
        self.face = face
        self.body = body
        self.arm = arm
        self.leg = leg

    def __str__(self):
        return "%s, %s, %s, %s" % (self.face, self.body, self.arm, self.leg)


class PlayerBuilder(metaclass=ABCMeta):
    @abstractmethod
    def build_face(self):
        pass

    @abstractmethod
    def build_body(self):
        pass

    @abstractmethod
    def build_arm(self):
        pass

    @abstractmethod
    def build_leg(self):
        pass


class GirlBuilder(PlayerBuilder):
    def __init__(self):
        self.player = Player()

    def build_face(self):
        self.player.face = "脸"

    def build_body(self):
        self.player.body = "身体"

    def build_arm(self):
        self.player.arm = "胳膊"

    def build_leg(self):
        self.player.leg = "腿"


class Monster(PlayerBuilder):
    def __init__(self):
        self.player = Player()

    def build_face(self):
        self.player.face = "怪兽脸"

    def build_body(self):
        self.player.body = "怪兽身材"

    def build_arm(self):
        self.player.arm = "长毛的胳膊"

    def build_leg(self):
        self.player.leg = "长毛的腿"


class PlayerDirector:  # 控制组装顺序
    def build_player(self, builder):
        builder.build_body()
        builder.build_face()
        builder.build_arm()
        builder.build_leg()
        return builder.player


# client

builder = Monster()
director = PlayerDirector()
p = director.build_player(builder)
print(p)
```

### 1.5 单例模式

* 内容：保证一个类只有一个实例，并提供一个访问它的全局访问点
* 角色：单例（Singleton）
* 优点
    * 对唯一实例的受控访问
    * 单例相当于全局变量，但防止了命名空间被污染

```python
class Singleton:
    def __new__(cls, *args, **kwargs):
        # 在 __init__ 方法之前被调用，用于分配空间、初始化对象等
        if not hasattr(cls, "_instance"):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


class MyClass(Singleton):
    def __init__(self, a):
        self.a = a


a = MyClass(10)
b = MyClass(20)
print(a.a)
print(b.a)
print(id(a), id(b))
"""output
20
20
1439396658816 1439396658816
"""
```

## 2 结构型模式

### 2.1 适配器模式

* 内容：将一个类的接口转换成客户希望的另一个接口。适配器模式使得原本由于接口不兼容而不能一起工作的那些类可以一起工作

* 两种实现方式
    * 类适配器：使用多继承
    * 对象适配器：使用组合
* 角色
    * 目标接口（Target）
    * 待适配的类（Adaptee）
    * 适配器（Adapter）
* 使用场景
    * 想使用一个已经存在的类，而它的接口不符合你的要求
    * （对象适配器）想使用一些已经存在的子类，但不可能对每一个都进行子类化以匹配他们的接口。对象适配器可以适配它的父类接口。

```python
from abc import ABCMeta, abstractmethod


class Payment(metaclass=ABCMeta):
    @abstractmethod
    def pay(self, money):
        pass


class Alipay(Payment):
    def pay(self, money):
        print("支付宝支付%d元。" % money)


class WechatPay(Payment):
    def pay(self, money):
        print("微信支付%d元。" % money)


class BankPay:
    def cost(self, money):
        print("银联支付%d元。" % money)


# 类适配器
# class NewBankPay(Payment, BankPay):
#     # 继承
#     def pay(self, money):
#         self.cost(money)

# 对象适配器
class PaymentAdapter:
    # 组合
    def __init__(self, payment):
        self.payment = payment

    def pay(self, money):
        self.payment.cost(money)


p = PaymentAdapter(BankPay())
p.pay(100)
```

### 2.2 桥模式

* 内容：将一个事物的两个维度分离，使其都可以独立地变化

* 角色
    * 抽象（Abstraction）
    * 细化抽象（RefinedAbstraction）
    * 实现者（Implementor）
    * 具体实现者（ConcreteImplementor）
* 应用场景：当事物有两个维度上的表现，两个维度都可能扩展时
* 优点
    * 抽象和实现相分离
    * 优秀的扩展能力

```python
from abc import ABCMeta, abstractmethod


class Shape(metaclass=ABCMeta):
    def __init__(self, color):
        self.color = color

    @abstractmethod
    def draw(self):
        pass


class Color(metaclass=ABCMeta):
    @abstractmethod
    def paint(self, shape):
        pass


class Rectangle(Shape):
    name = "长方形"

    def draw(self):
        # 长方形逻辑
        self.color.paint(self)


class Circle(Shape):
    name = "圆形"

    def draw(self):
        # 圆形逻辑
        self.color.paint(self)


class Red(Color):
    def paint(self, shape):
        print("红色的%s" % shape.name)


class Green(Color):
    def paint(self, shape):
        print("绿色的%s" % shape.name)


shape = Rectangle(Red())
shape.draw()
shape2 = Circle(Green())
shape2.draw()
```

### 2.3 组合模式

* 内容：将对象组合成树形结构以表示“部分-整体”的层次结构。组合模式使得用户对单个对象和组合对象的使用具有一致性。
* 角色
    * 抽象组件（Component）
    * 叶子组件（Leaf）
    * 复合组件（Composite）
    * 客户端（Client）
* 适用场景
    * 表示对象的“整体-部分”层次结构（特别是结构是递归的）
    * 希望用户忽略组合对象与单个对象的不同，用户统一地使用组合结构中的所有对象
* 优点
    * 定义了包含基本对象和组合对象的类层次结构
    * 简化客户端代码，即客户端可以一致地使用组合对象和单个对象
    * 更容易增加新类型的组件

```python
from abc import ABCMeta, abstractmethod


class Graphic(metaclass=ABCMeta):
    @abstractmethod
    def draw(self):
        pass


class Point(Graphic):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "点 (%s, %s)" % (self.x, self.y)

    def draw(self):
        print(str(self))


class Line(Graphic):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def __str__(self):
        return "线段[%s, %s]" % (self.p1, self.p2)

    def draw(self):
        print(str(self))


class Picture(Graphic):
    def __init__(self, iterable):
        self.children = []
        for g in iterable:
            self.add(g)

    def add(self, graphic):
        self.children.append(graphic)

    def draw(self):
        print("----------复合图形----------")
        for g in self.children:
            g.draw()
        print("----------复合图形----------")


p1 = Point(2, 3)
l1 = Line(Point(3, 4), Point(6, 7))
l2 = Line(Point(1, 5), Point(2, 8))
pic1 = Picture([p1, l1, l2])

pic1.draw()

"""output
----------复合图形----------
点 (2, 3)
线段[点 (3, 4), 点 (6, 7)]
线段[点 (1, 5), 点 (2, 8)]
----------复合图形----------
"""
```

### 2.4 外观模式

* 内容：为子系统中的一组接口提供一个一致的界面，外观模式定义了一个高层接口，这个接口使得这一子系统更加容易使用
* 角色
    * 外观（facade）
    * 子系统类（subsystem classes）
* 优点
    * 减少了系统相互依赖
    * 提高了灵活性
    * 提高了安全性

```python
class CPU:
    def run(self):
        print("CPU开始运行")

    def stop(self):
        print("CPU停止运行")


class Disk:
    def run(self):
        print("硬盘开始工作")

    def stop(self):
        print("硬盘停止工作")


class Memory:
    def run(self):
        print("内存通电")

    def stop(self):
        print("内存断电")


class Computer:
    def __init__(self):
        self.cpu = CPU()
        self.disk = Disk()
        self.memory = Memory()

    def run(self):
        self.cpu.run()
        self.disk.run()
        self.memory.run()

    def stop(self):
        self.cpu.stop()
        self.disk.stop()
        self.memory.stop()


computer = Computer()
computer.run()
computer.stop()
```

### 2.5 代理模式

* 内容：为其他对象提供一种代理以控制这个对象的访问
* 应用场景
    * 远程代理：为远程对象提供代理
    * 虚代理：根据需要创建很大的对象
    * 保护代理：控制对原始对象的访问，用于对象有不同访问权限时
* 角色
    * 抽象实体（Subject）
    * 实体（RealSubject）
    * 代理（Proxy）
* 优点
    * 远程代理：可以隐藏对象位于远程地址空间的事实
    * 虚代理：可以进行优化，例如根据要求创建对象
    * 保护代理：允许在访问一个对象时有一些附加的内务处理

```python
from abc import ABCMeta, abstractmethod


class Subject(metaclass=ABCMeta):
    @abstractmethod
    def get_content(self):
        pass

    @abstractmethod
    def set_content(self, content):
        pass


class RealSubject(Subject):
    def __init__(self, filename):
        self.filename = filename
        f = open(filename, 'r')
        print("读取文件内容")
        self.content = f.read()
        f.close()

    def get_content(self):
        return self.content

    def set_content(self, content):
        f = open(self.filename, 'w')
        f.write(content)
        f.close()


class VirtualProxy(Subject):
    def __init__(self, filename):
        self.filename = filename
        self.subj = None

    def get_content(self):
        if not self.subj:
            self.subj = RealSubject(self.filename)
        return self.subj.get_content()

    def set_content(self, content):
        if not self.subj:
            self.subj = RealSubject(self.filename)
        self.subj.set_content(content)


class ProtectedProxy(Subject):
    def __init__(self, filename):
        self.subj = RealSubject(filename)

    def get_content(self):
        return self.subj.get_content()

    def set_content(self, content):
        raise PermissionError("无写入权限")
```

## 3 行为型模式

### 3.1 责任链模式

* 内容：使多个对象都有机会处理请求，从而避免请求的发送者和接收者之间的耦合关系。将这些对象连成一条链，并沿着这条链传递该请求，直到有一个对象处理它为止
* 角色
    * 抽象处理者（Handler）
    * 具体处理者（ConcreteHandler）
    * 客户端（Client）
* 适用场景
    * 有多个对象可以处理一个请求，哪个对象处理由运行时决定
    * 在不明确接收者的情况下，向多个对象中的一个提交一个请求

* 优点
    * 降低耦合度：一个对象无需知道是其他哪一个对象处理请求


```python
from abc import ABCMeta, abstractmethod


class Handler(metaclass=ABCMeta):
    @abstractmethod
    def handle_leave(self, day):
        pass


class GeneralManager(Handler):
    def handle_leave(self, day):
        if day < 10:
            print("总经理准假%d" % day)
        else:
            print("你还是辞职吧")


class DepartmentManager(Handler):
    def __init__(self):
        self.next = GeneralManager()

    def handle_leave(self, day):
        if day <= 5:
            print("部门经理准假%s天" % day)
        else:
            print("部门经理职权不足")
            self.next.handle_leave(day)


class ProjectDirector(Handler):
    def __init__(self):
        self.next = DepartmentManager()

    def handle_leave(self, day):
        if day <= 1:
            print("项目主管准假%d天" % day)
        else:
            print("项目主管职权不足")
            self.next.handle_leave(day)
```

### 3.2 观察者模式

* 内容：定义对象间的一种一对多的依赖关系，当一个对象的状态发生改变时，所有依赖于它的对象都得到通知并被自动更新。观察者模式又称“发布-订阅”模式
* 角色
    * 抽象主题（Subject）
    * 具体主题（ConcreteSubject）—— 发布者
    * 抽象观察者（Observer）
    * 具体观察者（ConcreteObserver）—— 订阅者
* 适用场景
    * 当一个抽象模型有两个方面，其中一个方面依赖于另一个方面。将这两者封装在独立对象中，以使它们可以各自独立地改变和复用
    * 当对一个对象的改变需要同时改变其他对象，而不知道具体有多少对象有待改变
    * 当一个对象必须通知其他对象，而它又不能假定其它对象是谁。换言之，你不希望这些对象是紧密耦合的
* 优点
    * 目标和观察者之间的抽象耦合最小
    * 支持广播通信

```python
from abc import ABCMeta, abstractmethod


# 抽象订阅者
class Observer(metaclass=ABCMeta):
    @abstractmethod
    def update(self, notice):  # notice 是一个 Notice 类的对象
        pass


class Notice:  # 抽象发布者
    def __init__(self):
        self.observers = []

    def attach(self, obs):
        self.observers.append(obs)

    def detach(self, obs):
        self.observers.remove(obs)

    def notify(self):
        for obs in self.observers:
            obs.update(self)


class StaffNotice(Notice):
    def __init__(self, company_info=None):
        super().__init__()
        self.__company_info = company_info

    @property
    def company_info(self):
        return self.__company_info

    @company_info.setter
    def company_info(self, info):
        self.__company_info = info
        self.notify()


class Staff(Observer):
    def __init__(self):
        self.company_info = None

    def update(self, notice):
        self.company_info = notice.company_info


notice = StaffNotice("初始公司信息")
s1 = Staff()
s2 = Staff()
notice.attach(s1)
notice.attach(s2)
notice.company_info = "公司今天业绩非常好，给大家发奖金！！！"
print(s1.company_info)
```

### 3.3 策略模式

* 内容：定义一系列的算法，把它们一个个封装起来，并且使它们可相互替换。本模式使得算法可独立于使用它的客户而变化。
* 角色
    * 抽象策略（Strategy）
    * 具体策略（ConcreteStrategy）
    * 上下文（Context）
* 优点
    * 定义了一系列可重用的算法和行为
    * 消除了一些条件语句
    * 可以提供相同行为的不同实现
* 缺点
    * 客户必须了解不同的策略

```python
from abc import ABCMeta, abstractmethod


class Strategy(metaclass=ABCMeta):
    @abstractmethod
    def execute(self, data):
        pass


class FastStrategy(Strategy):
    def execute(self, data):
        print("用较快的策略处理%s" % data)


class SlowStrategy(Strategy):
    def execute(self, data):
        print("用较慢的策略处理%s" % data)


class Context:
    def __init__(self, strategy, data):
        self.strategy = strategy
        self.data = data

    def set_strategy(self, strategy):
        self.strategy = strategy

    def do_strategy(self):
        self.strategy.execute(self.data)


# Client

data = "[...]"
s1 = FastStrategy()
s2 = SlowStrategy()
context = Context(s1, data)
context.do_strategy()
context.set_strategy(s2)
context.do_strategy()
```

### 3.4 模板方法模式

* 内容：定义一个操作中的算法骨架，而将一些步骤延迟到子类中。模板方法使得子类可以不改变一个算法的结构，即可重定义该算法的某些特定步骤
* 角色
    * 抽象类（AbstractClass）：定义抽象的原子操作（钩子操作）；实现一个模板方法作为算法的骨架
    * 具体类（ConcreteClass）：实现原子操作
* 适用场景
    * 一次性实现一个算法的不变部分
    * 各个子类的公共行为应该被提取出来并集中到一个公共的父类中，以免代码重复
    * 控制子类扩展

```python
from abc import ABCMeta, abstractmethod
from time import sleep


class Window(metaclass=ABCMeta):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def repaint(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def run(self):  # 模板方法
        self.start()
        while True:
            try:
                self.repaint()
                sleep(1)
            except KeyboardInterrupt:
                break
        self.stop()


class MyWindow(Window):
    def __init__(self, msg):
        self.msg = msg

    def start(self):
        print("窗口开始运行")

    def stop(self):
        print("窗口结束运行")

    def repaint(self):
        print(self.msg)


MyWindow("Hello").run()
```

