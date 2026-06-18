import numpy as np

g0 = 9.80665
margin = 1.2

dv_europa = 1859.0
dry_europa = 694.96

dv_io = 1358.0
dry_io = 741.69


def size_prop(dry_mass, dv, isp=333.0):
    mr   = np.exp(dv / (isp * g0)) - 1
    prop = dry_mass * mr * margin
    wet  = dry_mass + prop
    return {"prop": prop, "wet": wet, "mr": mr}


if __name__ == "__main__":
    eu = size_prop(dry_europa, dv_europa)
    io = size_prop(dry_io,     dv_io)
    print(eu)
    print(io)