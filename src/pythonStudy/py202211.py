import sys
import ctypes
a=123
print(sys.getrefcount(a))
print(ctypes.c_long.from_address(id(a)).value)
b=456
print(sys.getrefcount(b))
print(ctypes.c_long.from_address(id(b)).value)

c=a
d=c
print(sys.getrefcount(a))
print(ctypes.c_long.from_address(id(a)).value)
print(sys.getrefcount(d))
print(ctypes.c_long.from_address(id(d)).value)

a_l = [1,2,3]
print(sys.getrefcount(a_l))
def funTestRef(x:list):
    x.append(4)
    print(sys.getrefcount(x))
funTestRef(a_l)
a_l
print(sys.getrefcount(a_l))

a_2=3
print(sys.getrefcount(a_2))
def funTestRef2(x:int):
    x=x+4
    print(sys.getrefcount(x))

funTestRef2(a_2)
print(sys.getrefcount(a_2))