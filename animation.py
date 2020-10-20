import numpy as np
import random as rand
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from itertools import combinations
from runtumble import RTP, Cell, Wall, Ball
from matplotlib.animation import FFMpegWriter

class FasterFFMpegWriter(FFMpegWriter):
    '''FFMpeg-pipe writer bypassing figure.savefig.'''
    def __init__(self, **kwargs):
        '''Initialize the Writer object and sets the default frame_format.'''
        super().__init__(**kwargs)
        self.frame_format = 'argb'

    def grab_frame(self, **savefig_kwargs):
        '''Grab the image information from the figure and save as a movie frame.

        Doesn't use savefig to be faster: savefig_kwargs will be ignored.
        '''
        try:
            # re-adjust the figure size and dpi in case it has been changed by the
            # user.  We must ensure that every frame is the same size or
            # the movie will not save correctly.
            self.fig.set_size_inches(self._w, self._h)
            self.fig.set_dpi(self.dpi)
            # Draw and save the frame as an argb string to the pipe sink
            self.fig.canvas.draw()
            self._frame_sink().write(self.fig.canvas.tostring_argb()) 
        except (RuntimeError, IOError) as e:
            out, err = self._proc.communicate()
            raise IOError('Error saving animation to file (cause: {0}) '
                      'Stdout: {1} StdError: {2}. It may help to re-run '
                      'with --verbose-debug.'.format(e, out, err)) 

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
        c.advance(0.05)
        print(f"{i:<3}: Right: {cell.right_flux}, Left: {cell.left_flux}", end="\r")
        return tuple(particle.draw(ax) for particle in particles)# + (w.draw(ax),)

    anim = animation.FuncAnimation(fig, animate, frames=300, interval=50, blit=True)

    if save:
        Writer = FasterFFMpegWriter # animation.writers['ffmpeg']
        writer = Writer(fps=24, bitrate=1800)
        anim.save('collision.mp4', writer=writer)
    else:
        plt.show()


c = Cell()
N = 25 # The number of RNT particles to simulate
p = []
for i in range(N):
    p.append(RTP())
for particle in p:
    particle.r = np.array((rand.random() * c.cell_size[0], rand.random() * c.cell_size[1]))
    c.add_particle(particle)
b = Ball(7.5,7.5,3,0.5)
c.add_particle(b)
p.append(b)

c.set_field(0) # The field strength
c.set_pbc(True) # Periodic boundary conditions
c.set_velocity(0.5) # Intrinsic particle velocity
c.set_decay_rate(0.5) # Particle decay rate
c.set_RTP_radius(0.1) # Particle radius -- relevant to avoid wall
                           # glitches at high velocity or field strength

do_animation(c,p,save=True)
