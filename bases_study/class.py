'''
1) class(클래스):  데이터와 데이터를 조작하는 함수의 묶음(큰 틀)
ex) 과자 틀(클래스) / 과자틀에 의해서 만들어진 과자(객체)
동일한 클래스로 만든 객체들은 서로 전혀 영향을 주지 않는다.

2) self : 클래스로 생성된 인스턴스 의미. = 객체 자기 자신을 참조하는 매개변수
ex) 붕어빵 틀(클래스)에 반죽을 넣어서 만들어진 붕어빵

* 클래스 내에 기재되어 있는 함수를 메소드라고도 부른다.
* 인스턴스: 객체가 어떤 클래스에 속할 때 그 객체를 그 클래스의 인스턴스라고 부른다.

* 이렇게 앞 뒤로 _ _(밑줄 두 개)가 붙은 메서드는 파이썬이 자동으로 호출해주는 메서드로
스페셜 메서드(special method) 또는 매직 메서드(magic method)라고 부른다.

* 메서드의 첫 번째 매개변수는 보통 self를 지정한다.

3) --init--: 인스턴스 초기화 할 때 불러오는 것(생성자)

4) --call--: 인스턴스가 호출됐을 때 실행

5) 상속: "재산을 상속받다"라고 할 때의 상속과 같은 의미
= 어떤 클래스를 만들 때 다른 클래스의 기능을 물려받을 수 있게 만드는 것

* 사용예제
1) class
# 무기 클래스 정의 
class Weapone(): pass 

#'무기' 클래스 객체 '칼'을 생성해보기"
sword = Weapone() print(type(sword))

2) init, self
class Person:
    def __init__(self, name, age, address):
        self.hello = '안녕하세요.'
        self.name = name
        self.age = age
        self.address = address
 
    def greeting(self):
        print('{0} 저는 {1}입니다.'.format(self.hello, self.name))
 
maria = Person('마리아', 20, '서울시 서초구 반포동')
maria.greeting()    # 안녕하세요. 저는 마리아입니다.
 
print('이름:', maria.name)       # 마리아
print('나이:', maria.age)        # 20
print('주소:', maria.address)    # 서울시 서초구 반포동
실행 결과
안녕하세요. 저는 마리아입니다.
이름: 마리아
나이: 20
주소: 서울시 서초구 반포동

###  Person이라는 클래스를 만들고 greeting 메서드 호출.
self 다음에 값을 받을 매개변수를 지정했다. 

3) init, self
class Person():
    def __init__(self,name,age,job):
        self.name = name
        self.age = age
        self.job = job

Steve = Person('Steve','22','programmer')

*self는 자기자신을 의미한다. 즉 인스턴스를 가리킨다.
*Person이 self에 들어갈거고 Person 클래스 인스턴스인 Steve를 통해 각 속성에 접근 할 수 있다.
*self 대신에 클래스 이름(Person)을 넣어도 상관없다.
*마찬가지로 name,age,job이 클래스 인자를 통해 전달되면 인스턴스의 속성이 생성된다.

4) init

>>> class FourCal:

    def __init__(self, first, second):

        self.first = first

        self.second = second

    def setdata(self, first, second):

        self.first = first

        self.second = second

    def add(self):

        result = self.first + self.second

        return result  

>>> a = FourCal(4, 2)

*위와 같이 수행하면 __init__ 메서드의 매개변수에는 각각 다음과 같은 값들이 대입
4
2

>>> a = FourCal(4, 2) 
>> a.add() 
6

5) call
class Plus :
    
    def __call__(self, x, y) :
        return x + y

plus = Plus()

n = plus(1, 2) # plus.__call__(1, 2)
print(n)       # 3
위 예제를 실행 시키면 plus 객체를 마치 함수 처럼 사용하여 3이라는 결과를 얻는다.

6) 상속
class Greet():
    greet = 'hello'
    introduct = 'My name is'
    
    def greeting(self):
        print('Nice to Meet you')
        
    def print_info(self):
        print(Greet.introduce, end = ' ')
        
class Profile(Greet):
    def __init__(self, name):
        self.name = name
        
if __name__ == '__main__':
    s1 = Profile('jihee')
    
    print(s1.greet)
    s1.print_info()
    s1.greeting()

* 상속하고자 하는 클래스명을 소괄호에 넣는다.
* 자식 클래스가 갖고 있는 않은 멤버라도 부모 클래스가 갖고 있다면 자식 클래스 객체를 통해 사용 가능
* 그러므로 자식 클래스 profile은 메서드 greeting, print_info를 가지고 있지 않지만 상속받았기에 사용 가능

7) class, self
class Singer:                      # 가수 정의
    def sing(self):                # 노래하기 메서드
       return "Lalala~"
           
taeji = Singer()                   # 태지를 만들어랏!
taeji.sing()                       # 노래 한 곡 부탁해요~
'Lalala~'

'''