import numpy as np
import random as rand
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from itertools import combinations
from runtumble import Particle, Cell, Wall


def do_animation(cell, particles, save=False):
    fig, ax = plt.subplots()
    for wall in cell.get_walls():
        wall.draw(ax)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-0.5,cell.cell_size[0] + 0.5)
    ax.set_ylim(-0.5,cell.cell_size[1] + 0.5)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    def animate(i):
        for particle in particles:
            particle.run(0.05)
        print(f"Right: {cell.right_flux}, Left: {cell.left_flux}", end="\r")
        return tuple(particle.draw(ax) for particle in particles)# + (w.draw(ax),)

    anim = animation.FuncAnimation(fig, animate, frames=1000, interval = 2, blit=True)

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=24, bitrate=1800)
        anim.save('collision.mp4', writer=writer)
    else:
        plt.show()


c = Cell()
N = 15 # The number of RNT particles to simulate
p = []
for i in range(N):
    p.append(Particle())
for particle in p:
    particle.set_r(rand.random() * c.cell_size[0], rand.random() * c.cell_size[1])
    c.add_particle(particle)

c.set_field(0.2) # The field strength
c.set_pbc(True) # Periodic boundary conditions
c.set_velocity(0.5) # Intrinsic particle velocity
c.set_decay_rate(0.5) # Particle decay rate
c.set_particle_radius(0.1) # Particle radius -- relevant to avoid wall
                           # glitches at high velocity or field strength

do_animation(c,p,save=False)
