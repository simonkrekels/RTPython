# Initialisation
## Cell
``` Cell(xsize=10, ysize=10, default_walls=False) ```
The Cell class serves as background for the simulation. It keeps track of particles and walls, as well as flux through the left-and right walls.

### Parameters
- ```xsize``` The size of the cell in the x-dimension. Default = 10.
- ```ysize``` The size of the cell in the y-dimension. Default = 10.
- ```default_walls``` If ```True```, the default walls will be loaded. If set to ```False```, the top-and bottom walls are loaded, but others are to be loaded manually. Default = False.

### Methods
- ```add_particle(particle), remove_particle(wall)``` Adds (removes) a particle to the cell.
- ```add_wall(wall), remove_wall(wall)``` Adds (removes) a wall to the cell.
- ```set_field(val)``` Sets field strength in the x-direction
- ```set_velocity(val)``` Sets intrinsic velocity for particles in cell
- ```set_decay_rate(val)``` Sets decay rate for particles in the cell
- ```set_pbc(True/False)``` Set whether boundaries are periodic
- ```set_particle_radius(val)``` Sets particle radius. Relevant to prevent wall glitches.

### Properties
- ```Cell.left_flux``` returns the number of particles which left the cell through the left boundary
- ```Cell.right_flux``` returns the number of particles which left the cell through the right boundary

## Wall
``` Wall(x0, y0, xf, yf) ```
A simple wall object defined as a straight line between (```x0```,```y0```) and (```xf```,```yf```)
### Parameters
```x0, y0, xf, yf``` are the x-y coordinates of the end points of the wall.

## Particle
``` Particle(x_pos=0, y_pos=0, angle=0)```
The Particle class models a RTP and manages decays and collisions with walls.
### Parameters
- ```x_pos``` The x_position of the particle. Default = 0.
- ```y_pos``` The y_position of the particle. Default = 0.
- ```angle``` The angle of the particle. Default = 0

### Methods
- ```run(dt)``` Advances the particle with a timestep ```dt```.
- ```tumble()``` Assigns a random new angle to the particle.
