import numpy as np
import matplotlib.pyplot as plt
#Europa
ae_EU = 1560.8
miu_EU = 3202.71210

#Io
ae_IO = 1821.6
miu_IO = 5959.9




def get_optimums(miu, ae, degrees = 30, plot = False):
    n = 1
    y = []
    x = []
    while n<degrees:
        r = np.arange(ae,ae+1000,1)
        Hopt = (ae/r) ** n*(miu/r ** 3) ** 0.5*(1-ae ** 2/r**2)
        target = (r-ae)
        if plot and (n % 5 == 0 or n == 1 or n == 2 or n == 30):
            plt.plot(target,Hopt, label=f"degree {n}")
            
        indices = np.where(Hopt == np.max(Hopt))[0][0]
        y.append(np.max(Hopt))
        x.append(target[indices])
        n=n+1
    if plot:
        plt.xlabel("altitude [km]")
        plt.ylabel("[-]")
        plt.title("Measurement efficiency against orbital height")
        plt.tight_layout()
        plt.legend()
        plt.show()
    
    return x, y

x1, y1 = get_optimums(miu_EU, ae_EU, degrees=31, plot=True)
x2, y2 = get_optimums(miu_IO, ae_IO)
fig, ax = plt.subplots()
ax.plot(np.arange(1,50),x1, label="Europa optimum altitude")
ax.plot(np.arange(1,50),x2, label="Io optimum altitude")

ax.axvline(x=8.0,  color='black', linestyle='--', linewidth=1, label='Io Static Gravity Field Requirement')
ax.axvline(x=30.0, color='gray',  linestyle=':',  linewidth=1, label='Europa Static Gravity Field Requirement')
ax.set_xlabel('harmonic degree [-]')
ax.set_ylabel('altitude [km]')
ax.set_title('Optimum Orbital Height For Static Gravity Field Determination')

# Legend and grid
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)


# plt.xlabel("harmonic degree [-]")
# plt.ylabel("altitude [km]")
# plt.legend()
plt.show()