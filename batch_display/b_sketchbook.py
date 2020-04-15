#%%


class hello:
    def __init__(self):
        self.a = 0
    def change_a(self,x):
        self.a = x
    def get_a(self):
        return self.a
#%%

N = 9
hh = []
for i in range(N):
    hh.append(hello())
    
#%%
for i in range(N):
    hh[i].change_a(i**2)

for i in range(N):
    print(hh[i].get_a())
    
#%%
import numpy as np
#%%

a = np.random.randint(0,4,size = (10,)

print(a)

#%%
a = 5
b = 2

assert a==b, (a+b, 3**3, "아무말")
print(a,b)
#%%

from b_sketchbook_temp import h2

hhh = h2(56)
print(hhh.get_c_1(2))


#%%


def h3():
    global HIHI
    HIHI = 7
    return HIHI
    
print(h3())
print(HIHI)

#%%


a = [[5,3],[5,3],[5,3],[5,3],[5,3]]
print(a[:][0])






