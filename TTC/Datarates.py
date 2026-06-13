EuEclipse = 0.811823
EuContact = 1.275248
EuOrbit = 2.087071


InstrumentRates = {
    "GALA" : 10,
    "3GM" : 1.5,
    "Telemetry" : 25
}   

ImagingData = {
    "Thermal" : 9.2 * 1024,
    "Imaging" : 25.17 * 1024
}

IoInstruments = {
    "GALA" : {"Generation": 200, "Downlink": 12 * 24 * 3600, "Storage" : 200},
    "3GM" : {"Generation": 200, "Downlink": 200, "Storage" : 200},
    "Telemetry" : {"Generation": 12*24*3600, "Downlink": 12*24*3600, "Storage" : 200}
}

EuInstruments = {
    "GALA" : {"Generation": EuOrbit * 3600, "Downlink": EuContact * 3600, "Storage" : EuOrbit * 3600},
    "3GM" : {"Generation": EuContact * 3600, "Downlink": EuContact * 3600, "Storage" : EuOrbit * 3600},
    "Telemetry" : {"Generation": EuOrbit * 3600, "Downlink": EuContact * 3600, "Storage" : EuOrbit * 3600}

}

IoImaging = {
    "Thermal" : {"NImages": 400, "Downlink" : 12*24*3600, "Generation" : 200},
    "Imaging" : {"NImages": 400, "Downlink" : 12*24*3600, "Generation" : 200}
}
EuImaging = {
    "Thermal" : {"NImages": 0, "Downlink" : EuContact * 3600, "Generation" : EuOrbit * 3600},
    "Imaging" : {"NImages": 320, "Downlink" : EuContact * 3600, "Generation" : EuOrbit * 3600}
}

IoStorage = {
    "GALA" : IoInstruments["GALA"]["Storage"]*InstrumentRates["GALA"],
    "3GM" : IoInstruments["3GM"]["Storage"]*InstrumentRates["3GM"],
    "Telemetry" : IoInstruments["Telemetry"]["Storage"]*InstrumentRates["Telemetry"],
    "Thermal" : IoImaging["Thermal"]["NImages"]*ImagingData["Thermal"],
    "Imaging" : IoImaging["Imaging"]["NImages"]*ImagingData["Imaging"]
}

EuStorage = {
    "GALA" : EuInstruments["GALA"]["Storage"]*InstrumentRates["GALA"],
    "3GM" : EuInstruments["3GM"]["Storage"]*InstrumentRates["3GM"],
    "Telemetry" : EuInstruments["Telemetry"]["Storage"]*InstrumentRates["Telemetry"],
    "Thermal" : EuImaging["Thermal"]["NImages"]*ImagingData["Thermal"],
    "Imaging" : EuImaging["Imaging"]["NImages"]*ImagingData["Imaging"]
}

EuGeneration = {
    "GALA" : EuInstruments["GALA"]["Generation"]*InstrumentRates["GALA"],
    "3GM" : EuInstruments["3GM"]["Generation"]*InstrumentRates["3GM"],
    "Telemetry" : EuInstruments["Telemetry"]["Generation"]*InstrumentRates["Telemetry"],
    "Thermal" : EuImaging["Thermal"]["NImages"]*ImagingData["Thermal"],
    "Imaging" : EuImaging["Imaging"]["NImages"]*ImagingData["Imaging"]
}

IoGeneration = {
    "GALA" : IoInstruments["GALA"]["Generation"]*InstrumentRates["GALA"],
    "3GM" : IoInstruments["3GM"]["Generation"]*InstrumentRates["3GM"],
    "Telemetry" : IoInstruments["Telemetry"]["Generation"]*InstrumentRates["Telemetry"],
    "Thermal" : IoImaging["Thermal"]["NImages"]*ImagingData["Thermal"],
    "Imaging" : IoImaging["Imaging"]["NImages"]*ImagingData["Imaging"]
}

EuGenRate = {
    "GALA" : InstrumentRates["GALA"],
    "3GM" : InstrumentRates["3GM"],
    "Telemetry" : InstrumentRates["Telemetry"],
    "Thermal" : EuGeneration["Thermal"]/EuImaging["Thermal"]["Generation"],
    "Imaging" : EuGeneration["Imaging"]/EuImaging["Imaging"]["Generation"]
}

IoGenRate = {
    "GALA" : InstrumentRates["GALA"],
    "3GM" : InstrumentRates["3GM"],
    "Telemetry" : InstrumentRates["Telemetry"],
    "Thermal" : IoGeneration["Thermal"]/IoImaging["Thermal"]["Generation"],
    "Imaging" : IoGeneration["Imaging"]/IoImaging["Imaging"]["Generation"]
}

IoDownlink = {
    "GALA" : IoGeneration["GALA"]/IoInstruments["GALA"]["Downlink"],
    "3GM" : IoGeneration["3GM"]/IoInstruments["3GM"]["Downlink"],
    "Telemetry" : IoGeneration["Telemetry"]/IoInstruments["Telemetry"]["Downlink"],
    "Thermal" : IoGeneration["Thermal"]/IoImaging["Thermal"]["Downlink"],
    "Imaging" : IoGeneration["Imaging"]/IoImaging["Imaging"]["Downlink"],
}

EuDownlink = {
    "GALA" : EuGeneration["GALA"]/EuInstruments["GALA"]["Downlink"],
    "3GM" : EuGeneration["3GM"]/EuInstruments["3GM"]["Downlink"],
    "Telemetry" : EuGeneration["Telemetry"]/EuInstruments["Telemetry"]["Downlink"],
    "Thermal" : EuGeneration["Thermal"]/EuImaging["Thermal"]["Downlink"],
    "Imaging" : EuGeneration["Imaging"]/EuImaging["Imaging"]["Downlink"],
}

print("Io Generation")
print(f"{IoGeneration}")
print(f"Total: {sum(IoGeneration[inst] for inst in IoGeneration)/1024/1025} Gbit\n")

print("Eu Generation")
print(f"{EuGeneration}")
print(f"total: {sum(EuGeneration[inst] for inst in EuGeneration)/1024/1025} Gbit\n")

print("Io Storage")
print(f"{IoStorage}")
print(f"Total: {sum(IoStorage[inst] for inst in IoStorage)/1024/1025} Gbit\n")

print("Eu Storage")
print(f"{EuStorage}")
print(f"total: {sum(EuStorage[inst] for inst in EuStorage)/1024/1025} Gbit\n")

print("Io Generation Rate")
print(f": {IoGenRate}")
print(f"Total: {sum(IoGenRate[inst] for inst in IoGenRate)/1024} Mbps \n")

print("Eu Generation Rate")
print(f": {EuGenRate}")
print(f"Total: {sum(EuGenRate[inst] for inst in EuGenRate)/1024} Mbps \n")

print("Io Downlink")
print(f"{IoDownlink}")
print(f"Total: {sum(IoDownlink[inst] for inst in IoDownlink)*1024} kbps \n")

print("Eu Downlink")
print(f"{EuDownlink}")
print(f"Total: {sum(EuDownlink[inst] for inst in EuDownlink)*1024} kbps \n")