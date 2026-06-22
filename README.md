# *Analysis and Simulation of Mission to Jupiter*

<img width="1316" height="873" alt="image" src="https://github.com/user-attachments/assets/8bc42260-4610-4ca6-8a1a-e1e15cb5a182" />

This repo contains all of the code a team of 10 students used for a preliminary study of a deep space mission to Jupiter's moons Io and Europa.
Primary trajectory calculations are performed with TUDat Space [![TUDat](https://img.shields.io/badge/TUDat-GitHub-blue)](https://github.com/tudat-team/tudatpy).

# TUDat Space / Kerbal Space program interface

For this preliminary design, the trajectory choice, especially for the flybys, has to be tuned very carefully. Unfortunaetly, at this stage TUDat Space lacks the means to visualize the trajectories efficiently, aside from the 
builtin python matplotlib support (which is very slow due to the inherent cpu-side rendering). 

Kerbal space proggram, on the other hand, can be modified to accomodate N-body simulations with integrators and propogators similar to TUDat Space, to enable real time tracing of the N-body trajectories. One such implementation is
Principia  [![TUDat](https://img.shields.io/badge/Principia-GitHub-blue)](https://github.com/mockingbirdnest/Principia), which can perform N-body simulations, while a RealSolarSystem add-on is used to generate the visuals of the 
Solar System  [![TUDat](https://img.shields.io/badge/RSS-GitHub-blue)](https://github.com/KSP-RO/RealSolarSystem/). 


Our approach proposes to "combine" TUDat Space and Kerbal Space Program, by using TUDat Space to perform the mathematical calculations, while leveraging the graphical aspects of KSP for trajectory visualizations and tuning. 
These two radically different approaches are combined by coupling the spacecraft / planet state ephemeris. 
For example, in the N-body KSP engine, the spacecraft ephemeris can be exported as:

```
ORBIT {
    SMA   = 18859035560.264233
    ECC   = 0.98804852543645449
    INC   = 89.900551367034282
    LPE   = 86.444023939868416
    LAN   = 106.37893435189432
    MNA   = 6.2809071433766599
    EPH   = 28507011.169143163
    REF   = 5
    IDENT = Jupiter
}
```
Which can then be fed directly into TUDat, where physical simulations are ran. Alternatively, the spacecraft can be visualy placed / tuned manualy for flybys, with the coresponding ephemeris then exported to 
TUDat. This allows for rapid, preliminary prototypinig, which is extremely useful for the underlying DSE, considering the limited amount of time and resources at our disposal. In the future, this could be expanded by including 
a visualization tool within TUDat (OpenGL or Vulcan based, etc).

Clearly, this synchronization between ksp and TUDat has to be done sufficiently frequently, to prevent error growth. Additionally, tiny corrections may be needed to the exported / imported ephemeris. 
These adjustments can be done by simply varying the true anomaly of the spacecraft, until the desired trajectory is reached. 

# Io trajectory simulations

To simulate mission segments of the Io trajectory, we employ the help of TUdat Space. 
## Orbital Insertion

## Transition between High eccentricity and Primary Science Phases

To showcase the transition between high eccentricity and primary science phases, the goal is to simulate a single segment showcasing the possibility of utilizing the Galilean moons for trajectory change.

The main idea of this simulation is to show that a flyby of three galilean moons is possible. These are picked to be Io, Ganymede and Callisto. 

# Europa Trajectory Simulation

## Pump Down 
The pumpdown phase, ideally, should be an almost ballistic transfer between the initial orbit to which the spacecraft arrives to the target orbit around Ganymede, from which the spacecraft is then expected to ballistically transfer 
to Europa. For the current implementation, however, this is not feasible, as the pump-down phase is expected to contain up to 30 flybys of the Galilean moons, the simulation of which would require extreme precison, coupled with 
incredible requirement for computational resources. To this end, to showcase the potential of the ballistic pump-down phase, only a segment of this phase is simulated. 

<img width="1068" height="865" alt="image" src="https://github.com/user-attachments/assets/6efd026f-9397-43f5-b395-edafd7985962" />

The setup given simulates 7 flybys of the Gallilean moons; For this case, all of them involve Ganymede. As can be clearly seen, just 7 flybys is enough to reduce the orbital energy substantially. Also, as can be seen, the flybys
tend to shift the perijove closer towards Jupiter. This is undesirable, as for the final orbit it is desirable to have the perijove no lower than that of Europa (as, otherwise, the ballistic capture at Europa may become infeasible).
In the figure, the yellow-purple-blue orhoganal axes represent correction burns. These have been tuned manualy (by hand) to produce the given trajectory. This is due to the time constrains for the project. In practice, this could
be optimized in a way to completely ommit these deterministic correction burns. The total delta V required for the given trajectory is only 20m/s. Consider, that an impulsive burn of more than 2km/s would be required to make the change
without gravity assists. Thereby, this clearly demonstrates the excellent possibility to utilize the galilean moons for the pump-down phase. 

## Ballistic Capture
The most challenging 


The following results are obtained:
  <td>
      <img  src="https://github.com/user-attachments/assets/088e03b3-0562-4005-b264-754c55f7dc4b" 
 width="100%" >
        <figcaption align="center"> Weak Ballistic Capture Around Europa from an orbit originating at one of the Ganymedes Invariant Manifolds. </figcaption>
    </td>

As can be seen, the initial trajectory (although resembling that of a Hohman transfer, reachable by L2 invariant manifold from Ganymede over a larger timescale ( ~15 orbits))
gets weakly-captured by Europa. Improtantly to emphasize, this trajectory is purely ballistic. No correction burns are performed. However, the weak capture orbit around Europa is very large; this is a result of a manifold mismatch
not placing the spacecraft into correct injection orbit. This can be visualized in a rotating / synoidic frame:

  <td>
    <img src="https://github.com/user-attachments/assets/2969024c-deb4-45a7-b4ea-d415728c02ce" 
 width="100%" >
       <figcaption align="center">  </figcaption>
    </td>

As can be seen, this weakly captured orbit is of little use, as it does not approach Europa sufficiently close. 
This can be aleviated by sufficiently adjusting the manifold-manifold transfer. In practice, this would involve small correction burns, in the current implementation, this is achieved by slightly adjusting the initial parameters
of the L2 - bound weakly stable orbit around Ganymede. The Weakly Capture at Europa then takes the form, visualized in Europa - Centered Inertial Frame:
    
<table>
  <tr>
     <td>
    <img src="https://github.com/user-attachments/assets/a1344087-a18f-4832-b308-90bf6116672e" 
 width="100%" >
        <figcaption align="center"> Weak capture at Europa without an injection burn. For this configuration, the weak capture eventually results in an impact. </figcaption>
    </td>
    <td>
     <img src="https://github.com/user-attachments/assets/e87f8fc9-1d0e-4731-b724-143ea651015c" 
 width="100%">
       <figcaption align="center"> Weak capture at Europa with a 770m/s injection burn. For this configuration, the spacecraft enters a polar Europa orbit. </figcaption>
    </td>
  </tr>
</table>

The corresponding trajectory in the rotating Europa - Jupiter potential frame:


<td>
     <img  src="https://github.com/user-attachments/assets/12c31a27-c380-47ea-b310-42cbe4f4df77" 
 width="100%" >
        <figcaption align="center"> </figcaption>
</td>

While this trajectory simulated is not a full on trajectory from Jupiter Orbiter Insertion, it is a good demonstrator that, in detail design, these trajectories could be feasible, and may be merged together into a full mission plan. 

