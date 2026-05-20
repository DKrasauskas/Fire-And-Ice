import numpy as np
import matplotlib.pyplot as plt

##########################
#Hi this code was made by me, Witek. It calculates the optimal orbit height for getting static gravity.
# formula found in this paper
#Sośnica, K., “Orbit design for a future geodetic satellite and gravity field recovery,” Journal of Geodesy,
# Vol. 98, No. 77, 2024. https://doi.org/10.1007/s00190-024-01884-9.
##########################

#Europa
ae = 1560800 #body radius
miu = 3202.71210e6 # gravitational parameter
#Io
ae = 1821600
#Not finished

n = 1
y = []
x = []
while n<50:
    r = np.arange(ae,ae+500000,1000)
    Hopt = (ae/r)**n*(miu/r**3)**0.5*(1-ae**2/r**2)
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
