import math
import numpy as np
import matplotlib.pyplot as plt

DataEu = 20314.49304
DataIo = 11944.22857

C = 299792458
Au = 149597871000
dist = 6.2*Au

EbN0 = 5.7

antennaMassRatio = 4
systmeMass = 21

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
        EIRP = EbN0 - Ls - self.GS.GainTemp(self.frequency) + -228.6 + 10 * math.log10(self.dataRate)
        return EIRP
    
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
        antennaMass = antennaArea * antennaMassRatio
        totalMass = systmeMass + antennaMass

        return diameter, antennaArea, antennaMass, totalMass



ESADSN = GroundStation(diameter=35, temp=11.15, efficiency=0.7)
print(ESADSN.GainTemp(12))

EuMain = Antenna(dataRate=DataEu, frequency=12, GS=ESADSN, efficiency=0.55)
IoMain = Antenna(dataRate=DataIo, frequency=12, GS=ESADSN, efficiency=0.55)

fig, ax = plt.subplots()
powers = np.arange(1, 50, .1)
ax.plot(powers, [EuMain.AntennaSizing(power)[3] for power in powers], label="Europa")
ax.plot(powers, [IoMain.AntennaSizing(power)[3] for power in powers], label="Io")
ax.set_xlabel("Signal Power (W)")
ax.set_ylabel("TT&C Mass (kg)")
ax.legend()
plt.show()