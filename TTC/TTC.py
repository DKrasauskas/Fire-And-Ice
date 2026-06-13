import math
import numpy as np
import matplotlib.pyplot as plt

DataEuTele = 3*1024
DataEuSec = 1899844.926556508

DataIoTele = 3*1024
DataIoSec = 41042.125432098765

C = 299792458
Au = 149597871000
dist = 6.2*Au

# EbN0 = 1.2
EbN0 = 1.2
# EbN0 = 2
antennaMassRatio = 2.652582385
antennaFixed = 1.312 + 2.5
systmeMass = 26.2

class GroundStation:
    def __init__(self, diameter, temp, efficiency):
        self.diameter = diameter
        self.temp = temp
        self.efficiency = efficiency

    def GainTemp(self, frequency):
        frequency = frequency * 10**9
        Lamda = C / frequency
        gain = (math.pi * self.diameter / Lamda) ** 2 * self.efficiency
        gainDb = 10 * math.log10(gain)
        tempDb = 10 * math.log10(self.temp)
        GainTemp = gainDb - tempDb
        return GainTemp
    
class Antenna:
    def __init__(self, dataRate, frequency, GS, efficiency):
        self.dataRate = dataRate
        self.frequency = frequency
        self.GS = GS
        self.efficiency = efficiency

    def reqEirp(self):
        Lamda = C / (self.frequency * 10**9)
        Ls = 20 * math.log10(Lamda/(4 * math.pi * dist))
        EIRP = EbN0 - Ls - self.GS.GainTemp(self.frequency) -228.6 + 10 * math.log10(self.dataRate)
        return EIRP + 1.5
    
    def reqGain(self, signalPower):
        EIRP = self.reqEirp()
        PowerDb = 10 * math.log10(signalPower)
        Gain = EIRP - PowerDb
        return Gain
    
    def AntennaSizing(self, signalPower):
        Gain = self.reqGain(signalPower)
        gainLinear = 10 ** (Gain / 10)
        Lamda = C / (self.frequency * 10**9)
        diameter = math.sqrt(gainLinear * (Lamda / math.pi) ** 2 / self.efficiency)
        
        antennaArea = math.pi * (diameter / 2) ** 2
        antennaMass = antennaArea * antennaMassRatio + antennaFixed
        totalMass = systmeMass + antennaMass

        return diameter, antennaArea, antennaMass, totalMass



ESADSN = GroundStation(diameter=35, temp=11.15, efficiency=0.7)

EuMain = Antenna(dataRate=DataEuTele, frequency=8.4, GS=ESADSN, efficiency=0.55)
IoMain = Antenna(dataRate=DataIoTele, frequency=8.4, GS=ESADSN, efficiency=0.55)

EuMainSec = Antenna(dataRate=DataEuSec, frequency=32, GS=ESADSN, efficiency=0.55)
IoMainSec = Antenna(dataRate=DataIoSec, frequency=32, GS=ESADSN, efficiency=0.55)

fig, ax = plt.subplots()
powers = np.arange(1, 50, .1)
ax.plot(powers, [EuMain.AntennaSizing(power)[0] for power in powers], label="Europa Safe")
ax.plot(powers, [IoMain.AntennaSizing(power)[0] for power in powers], label="Io Safe")
ax.plot(powers, [EuMainSec.AntennaSizing(power)[0] for power in powers], label="Europa")
ax.plot(powers, [IoMainSec.AntennaSizing(power)[0] for power in powers], label="Io")
ax.set_xlabel("Signal Power (W)")
ax.set_ylabel("Diameter [m]")
ax.legend()
plt.show()

print(f"Europa Diameter, Area, Antenna mass and total mass: {EuMainSec.AntennaSizing(35)}")
print(f"Io Diameter, Area, Antenna mass and total mass: {IoMainSec.AntennaSizing(35)}")

print(f"Europa Safe Diameter, Area, Antenna mass and total mass: {EuMain.AntennaSizing(35)}")
print(f"Io Safe Diameter, Area, Antenna mass and total mass: {IoMain.AntennaSizing(35)}")

print((math.sqrt(0.5/12)*70*(C/(32*10**9)))/2.01)