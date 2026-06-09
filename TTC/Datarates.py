EuEclipse = 0.811823
EuContact = 1.275248
EuOrbit = 2.087071


InstrumentRates = {
    "GALA" : 10,
    "3GM" : 1.5,
    "Telemetry" : 2
}   

InstrumentDataSize = {
    "Thermal" : 9.2 * 1024,
    "Imaging" : 25.17 * 1024
}

#gala for flybys will only make data for abbout 2 minutes, afterwards it can send data
#over 50 days
#3GM will only sent data for abbout 2 minutes, and has to be send during those 2 minutes.

IoRates = {
    # "GALA" : {"OperationTime": 2 * 60, "DataReturnTime": 50 * 24 * 3600, "DataStorageTime" : 2 * 60},
    # "3GM" : {"OperationTime": 2*60, "DataReturnTime": 2*60, "DataStorageTime" : 0},
    "Telemetry" : {"OperationTime": 1, "DataReturnTime": 1, "DataStorageTime" : 0}
}

EuRates = {
    # "GALA" : {"OperationTime": EuOrbit * 3600, "DataReturnTime": EuContact * 3600, "DataStorageTime" : EuEclipse * 3600},
    # "3GM" : {"OperationTime": EuContact * 3600, "DataReturnTime": EuContact * 3600, "DataStorageTime" : 0},
    "Telemetry" : {"OperationTime": EuOrbit * 3600, "DataReturnTime": EuContact * 3600, "DataStorageTime" : EuEclipse * 3600}

}


IoData = {
    # "Thermal" : {"NImages": 24, "ReturnTime" : 50*24*3600},
    # "Imaging" : {"NImages": 24, "ReturnTime" : EuContact * 3600}
}
EuData = {
    # "Thermal" : {"NImages": 0, "ReturnTime" : 50*24*3600},
    # "Imaging" : {"NImages": 320, "ReturnTime" : EuContact * 3600}
}
IoDatarate = sum([InstrumentRates[inst]*IoRates[inst]["OperationTime"]/IoRates[inst]["DataReturnTime"] for inst in IoRates])
EuDatarate = sum([InstrumentRates[inst]*EuRates[inst]["OperationTime"]/EuRates[inst]["DataReturnTime"] for inst in EuRates])

# print(f"Data rate for Io: {IoDatarate} kbps")
# print(f"Data rate for Europa: {EuDatarate } kbps \n")

print(f"Io data stored: {sum([InstrumentRates[inst]*IoRates[inst]["OperationTime"] for inst in IoRates])/1024} mb")
print(f"Eu data stored: {sum([InstrumentRates[inst]*EuRates[inst]["OperationTime"] for inst in EuRates])/1024} mb \n")


IoImagingDatarate = sum([InstrumentDataSize[inst]*IoData[inst]["NImages"]/IoData[inst]["ReturnTime"] for inst in IoData])
EuImagingDatarate = sum([InstrumentDataSize[inst]*EuData[inst]["NImages"]/EuData[inst]["ReturnTime"] for inst in EuData])

IoImagingStorage = sum([InstrumentDataSize[inst]*IoData[inst]["NImages"] for inst in IoData])
EuImagingStorage = sum([InstrumentDataSize[inst]*EuData[inst]["NImages"] for inst in EuData])
# print(f"Data rate for Io imaging: {IoImagingDatarate} kbps\n")
# print(f"Data rate for EU imaging: {EuImagingDatarate} kbps\n")
# print(f"Data storage Io = {IoImagingStorage/1024}")
# print(f"Data storage Eu = {EuImagingStorage/1024}\n")

# print(f"Only science for europa = {EuDatarate}")
print(f"Total europa = {(EuImagingDatarate+EuDatarate)*1024}\n")

print(f"Only imaging for Io = {IoImagingDatarate}")
print(f"Total io = {(IoImagingDatarate+IoDatarate)*1024}\n")