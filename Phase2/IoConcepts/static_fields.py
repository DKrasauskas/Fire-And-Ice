import numpy as np
import matplotlib.pyplot as plt
#Europa
ae = 1560800
miu = 3202.71210e6

#Io
ae = 1821600
miu = 6.67e-11 * 8.931e22

n = 1
y = []
x = []
while n<8:
    r = np.arange(ae,ae+500000,1000)
    Hopt = (ae/r) ** n*(miu/r ** 3) ** 0.5*(1-ae ** 2/r**2)
    print(r[np.argmax(Hopt)]-ae)
    target = (r-ae)/1000
    plt.plot(target,Hopt)
    indices = np.where(Hopt == np.max(Hopt))[0]
    print(indices)
    y.append(np.max(Hopt))
    x.append(target[indices])
    n=n+1
plt.plot(x, y)
plt.show()

plt.figure()
plt.scatter(np.arange(1,50),x)
plt.show()