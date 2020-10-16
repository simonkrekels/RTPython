import matplotlib.pyplot as plt
from math import asin, acos, atan2, sin, cos, sqrt, pi
import numpy as np
import random as rand
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

"""Some simple 2-vector operations redefined.
These simple operations seem to be faster than NumPy
equivalents, maybe due to Numpy overhead"""

def norm(array):
    return (array[0]**2 + array[1]**2)**0.5

def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

class Wall():
    """A class defining a wall and some properties.
    A wall is defined by its endpoints and is a straight line between them."""
    THICKNESS = 1.5 # Thickness is only important for visual representation

    def __init__(self, x0, y0, xf, yf):
        self.r0 = np.array((x0, y0)) # Set start-and endpoints of wall (r0 anf rf resp.)
        self.rf = np.array((xf, yf))
        self.length = self.wall_length() # Calculate wall length and store for fast access
        self.vector = self.wall_vector() # Calculate wall vector and store for fast access
        self.angle = self.wall_angle() # Calculate wall angle and store for fast access

        if self.angle in [0, pi]: # Store whether the wall is horizontal or vertical (or neither)
            self.horizontal = True # Horiz. and vert. walls have simpler distance_to() methods
            self.vertical = False
        elif self.angle in [pi/2, 3*pi/2]:
            self.horizontal = False
            self.vertical = True
        else:
            self.horizontal = False
            self.vertical = False

        self._cell = None

    def __str__(self):
        """String representation of a wall: display start-and endpoints"""
        return f"Wall: {self.r0} -- {self.rf}"

    @property
    def cell(self):
        """reference to the cell the wall is in. Needed to implement wall movement"""
        return self._cell

    @cell.setter
    def cell(self, c):
        self._cell = c

    def distance_to(self, particle):
        """Returns the distance from the wall to a particle."""
        if self.horizontal:
            comp_x = [self.r0[0] - particle.x, self.rf[0] - particle.x]
            if all(i >= 0 for i in comp_x): # particle to the left of wall
                return sqrt(min(comp_x)**2 + (particle.y - self.r0[1])**2)
            elif all(i <= 0 for i in comp_x): # particle to the right of wall
                return sqrt(max(comp_x)**2 + (particle.y - self.r0[1])**2)
            else: # particle between endpoints
                return abs(particle.y - self.r0[1])
        elif self.vertical:
            comp_y = [self.r0[1] - particle.y, self.rf[1] - particle.y]
            if all(j <= 0 for j in comp_y): # particle above wall 
                return sqrt(max(comp_y)**2 + (particle.x - self.r0[0])**2)
            elif all(j >= 0 for j in comp_y): # particle below wall
                return sqrt(min(comp_y)**2 + (particle.x - self.r0[0])**2)
            else: # particle between endpoints
                return abs(particle.x - self.r0[0])
        else:
            rp = particle.get_r()
            if all(self.r0 == rp) or all(self.rf == rp): # Particle on wall
                return 0
            if acos(dot((rp - self.r0) / norm(rp - self.r0),
                        (self.rf - self.r0) / norm(self.rf - self.r0))) > pi / 2:
                # particle 'next' to wall, closest to r0
                return norm(rp - self.r0)
            if acos(dot((rp - self.rf) / norm(rp - self.rf),
                        (self.r0 - self.rf) / norm(self.r0 - self.rf))) > pi / 2:
                # Particle next to wall, closest point is rf
                return norm(rp - self.rf)
            else:
                # Particle inbetween endpoints, calculate perpendicular distance to line
                return abs(cross(self.r0 - self.rf, self.r0 - rp)) / norm(self.rf - self.r0)

    def wall_length(self):
        """Euclidean length of the wall"""
        return norm(self.rf - self.r0)

    def wall_vector(self):
        """Returns the vector from r0 to rf"""
        return self.rf - self.r0

    def wall_angle(self):
        """returns the angle between the wall_vector and the x-axis"""
        t = acos(dot(self.vector, np.array((1,0))) / norm(self.vector))
        s = asin(cross(self.vector, np.array((0,1))) / norm(self.vector))
        if s <= 0:
            return t
        else:
            return -t

    def update(self):
        """Update the stored wall angle, vector and length when changed by dynamics"""
        self.angle = self.wall_angle()
        if self.angle in [0, pi]:
            self.horizontal = True
            self.vertical = False
        elif self.angle in [pi/2, 3*pi/2]:
            self.horizontal = True
            self.vertical = False
        self.length = self.wall_length()
        self.vector = self.wall_vector()

    def rotate(self, angle, pos='r0'):
        """Rotate the wall by angle around pos."""
        if pos not in ['r0', 'rf', 'mid']:
            raise ValueError(f"Wall cannot rotate around {pos}")
        l = self.wall_length()
        t = self.wall_angle()
        if pos == 'r0':
            rfnew = self.r0 + np.array((l * cos(t - angle), l * sin(t - angle)))
            # check if new wall is within cell boundaries
            if rfnew[0] <= self.cell.cell_size[0] and rfnew[1] <= self.cell.cell_size[1]:
                self.rf = rfnew
        elif pos == 'rf':
            r0new = self.rf - np.array((l * cos(t - angle), l * sin(t - angle)))
            # check if new wall is within cell boundaries
            if r0new[0] <= self.cell.cell_size[0] and r0new[1] <= self.cell.cell_size[1]:
                self.r0 = r0new
        elif pos == 'mid':
            rmid = (self.r0 + self.rf)/2
            rfnew = rmid + np.array((l * cos(t - angle), l * sin(t - angle)))
            r0new = rmid - np.array((l * cos(t - angle), l * sin(t - angle)))
            # check if new wall is within cell boundaries
            if (rfnew[0] <= self.cell.cell_size[0] and rfnew[1] <= self.cell.cell_size[1] and
                    r0new[0] <= self.cell.cell_size[0] and r0new[1] <= self.cell.cell_size[1]):
                self.r0 = r0new
                self.rf = rfnew
        self.update()

    def translate(self, dx, dy):
        """Translate the entire wall (both r0 and rf) by dx and dy"""
        dr = np.array((dx, dy))
        r0new = self.r0 + dr
        rfnew = self.rf + dr
        # check if new wall is within cell boundaries
        if (rfnew[0] <= self.cell.cell_size[0] and rfnew[1] <= self.cell.cell_size[1] and
                    r0new[0] <= self.cell.cell_size[0] and r0new[1] <= self.cell.cell_size[1]):
            self.r0 = r0new
            self.rf = rfnew
        self.update()

    def draw(self, ax):
        """Graphical representation of wall"""
        # ax.plot([self.r0[0], self.rf[0]], [self.r0[1], self.rf[1]], color='black', linewidth=self.THICKNESS)
        line = Line2D([self.r0[0], self.rf[0]], [self.r0[1], self.rf[1]], color='black', linewidth=self.THICKNESS)
        ax.add_line(line)
        return line

class Cell():
    """A class for a cell in which simulations take place."""
    walls = []
    particles = []
    velocity = 0.5 # RTP intrinsic velocity
    decay_rate = 0.5 # RTP decay rate
    RTP_radius = 0.15
    field = 0 # Field acting on RTP's
    pbc = False # Periodic boundary conditions

    def __init__(self, xsize=10, ysize=10, particles=[], default_walls=True):
        self.cell_size = (xsize, ysize)
        particles = []
        self.add_wall(Wall(0, ysize, xsize, ysize)) # Top and bottom walls always included
        self.add_wall(Wall(0, 0, xsize, 0))
        if default_walls: # default snailhouse wall shape
            self.add_wall(Wall(0, 0, 0, ysize/2))
            self.add_wall(Wall(xsize, 0, xsize, ysize/2))
            self.add_wall(Wall(xsize, ysize/2, xsize/2, ysize/2))
        for p in particles: # add particles if any are specified
            self.add_particle(p)
        self._left_flux = 0 # Left-and right tally of particles passing boundary 
        self._right_flux = 0 # Only when pbc=True

    @property
    def left_flux(self):
        """Keeps a tally of particles travelling through left wall"""
        return self._left_flux

    @left_flux.setter
    def left_flux(self, value):
        self._left_flux = value

    @property
    def right_flux(self):
        """Keeps a tally of particles travelling through right wall"""
        return self._right_flux

    @right_flux.setter
    def right_flux(self, value):
        self._right_flux = value

    def update_cells(self):
        """update data in particles when cell data is changed"""
        for p in self.get_particles():
            p.update_cell()

    def set_pbc(self, value):
        self.pbc = value
        self.update_cells()

    def get_pbc(self):
        return self.pbc

    def add_particle(self, particle):
        if particle not in self.particles:
            self.particles.append(particle)
            particle.cell = self

    def set_velocity(self, velocity):
        self.velocity = velocity
        self.update_cells()

    def get_velocity(self):
        return self.velocity

    def set_decay_rate(self, rate):
        self.decay_rate = rate
        self.update_cells()

    def get_decay_rate(self):
        return self.decay_rate

    def set_RTP_radius(self, radius):
        self.RTP_radius = radius
        self.update_cells()

    def get_RTP_radius(self):
        return self.RTP_radius

    def set_field(self, field):
        self.field = field
        self.update_cells()

    def get_field(self):
        return self.field

    def remove_particle(self, particle):
        if particle in self.particles:
            del particle.cell
            self.particles.remove(particle)

    def add_wall(self, wall):
        if wall not in self.walls:
            self.walls.append(wall)
        wall.cell = self

    def remove_wall(self, wall):
        self.walls.remove(wall)

    def get_walls(self):
        return self.walls

    def get_particles(self):
        return self.particles

class Particle():
    """A particle class to model particles and moving objects. Particle class is meant to be abstract."""
    def __init__(self, x_pos=0, y_pos=0, cell=None):
        self._x = x_pos
        self._y = y_pos
        self._cell = cell
        self._periodic = False

    @property
    def x(self):
        """The x-coordinate"""
        return self._x

    @x.setter
    def x(self, value):
        if self._periodic: # periodic bc's means position modulo cell size
            if value > self.cell.cell_size[0]:
                self.cell.right_flux += 1 # Update tally when particles travel through boundary
            elif value < 0:
                self.cell.left_flux += 1 # Update tally when particles travel through boundary
            self._x = value % self.cell.cell_size[0]
        else:
            self._x = value

    @property
    def y(self):
        """The y-coordinate"""
        return self._y

    @y.setter
    def y(self, value): # cells are never periodic in y
        self._y = value

    def get_r(self):
        """return an array of coordinates. Bad design?"""
        return np.array((self.x, self.y))

    def set_r(self, x, y):
        """set x and y at once."""
        self.x = x
        self.y = y

    def set_r_vec(self, r):
        """set x and y at once by passing an array"""
        if len(r) != 2:
            raise TypeError
        self.set_r(r[0], r[1])

    @property
    def cell(self):
        """The cell this particle exists in"""
        return self._cell

    @cell.setter
    def cell(self, obj):
        if obj is not None:
            self._periodic = obj.get_pbc()
            self._cell = obj

    @cell.deleter
    def cell(self):
        return self._cell

    def update_cell(self):
        """Update particle parameters to match cell parameters"""
        if self.cell is not None:
            self._periodic = self.cell.get_pbc()

    @property
    def v(self):
        """The particle's current velocity"""
        return self._v

    @v.setter
    def v(self, value):
        self._v = value

    def get_v_vec(self):
        """Return a velocity vector"""
        return np.array((self.v * cos(self.moving_angle), self.v * sin(self.moving_angle)))

    @property
    def moving_angle(self):
        """The angle the particle is moving in"""
        return self._moving_angle

    @moving_angle.setter
    def moving_angle(self, value):
        self._moving_angle = value
        if self.cell:
            self.walls_on_path = self.get_walls_on_path()

    def draw(self, ax):
        """graphical representation as a circle"""
        circle = Circle(xy=self.get_r(), radius=self.RADIUS)
        ax.add_patch(circle)
        return circle

class RTP(Particle):
    '''A particle which models a run-and-tumble particle'''
    VELOCITY = 0.5 # RTP's intrinsic velocity
    RADIUS = 0.1 # RTP radius
    RATE = 5 # RTP decay rate
    FIELD = 0 # Field acting on RTP
    walls_on_path = [] # Walls this particle will or might collide with

    def __init__(self, x_pos=0, y_pos=0, angle=0, cell=None, pbc=False):
        super().__init__(x_pos, y_pos, cell)
        self._angle = angle
        self._time = 0
        self._runtime = rand.expovariate(self.RATE)
        self._v = self.VELOCITY
        self.moving_angle = self.free_moving_angle()
        self._periodic = pbc
        self._hovering_wall = None

    @property
    def angle(self):
        """The RTP's intrinsic angle"""
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value

    @property
    def time(self):
        """The RTP's internal clock"""
        return self._time

    @time.setter
    def time(self, value):
        self._time = value

    @property
    def runtime(self):
        """The RTP's lifetime. When RTP.time exceeds RTP.runtime,
        the particle tumbles"""
        return self._runtime

    @runtime.setter
    def runtime(self, value):
        self._runtime = value

    @property
    def hovering_wall(self):
        """A reference to the wall the RTP is hovering by. This wall
        will generally not be included in get_walls_on_path, but is liable to collisions"""
        return self._hovering_wall

    @hovering_wall.setter
    def hovering_wall(self, wall):
        self._hovering_wall = wall

    @Particle.cell.setter
    def cell(self, obj):
        """An override of Particle's cell.setter. RTP's need more
        data from cells than general particles"""
        if obj is not None:
            self.VELOCITY = obj.get_velocity()
            self.RADIUS = obj.get_RTP_radius()
            self.RATE = obj.get_decay_rate()
            self.FIELD = obj.get_field()
            self._periodic = obj.get_pbc()
            self._cell = obj

    def update_cell(self):
        """Update RTP's data to match cell's data"""
        if self.cell is not None:
            self.VELOCITY = self.cell.get_velocity()
            self.RADIUS = self.cell.get_RTP_radius()
            self.RATE = self.cell.get_decay_rate()
            self.FIELD = self.cell.get_field()
            self._periodic = self.cell.get_pbc()

    def free_moving_angle(self):
        """retrns the moving angle the particle would have when there
        are no walls nearby."""
        return atan2(self.VELOCITY * sin(self.angle),
                     (self.VELOCITY * cos(self.angle) + self.FIELD))

    def is_following_wall(self):
        """Check whether the RTP is following a wall"""
        if self.hovering_wall and self.hovering_wall.distance_to(self) < 3 * self.RADIUS:
            return True
        else:
            return False

    def get_v_vec(self):
        """return a velocity vector"""
        if self.hovering_wall:
            return np.array((self.v * cos(self.moving_angle) + self.FIELD,
                             self.v * sin(self.moving_angle)))
        else:
            v_f = sqrt((self.v * cos(self.angle) + self.FIELD)**2 + (self.v * sin(self.angle))**2)
            return np.array((v_f * cos(self.moving_angle), v_f * sin(self.moving_angle)))


    def run(self, dt=1):
        """Make the particle run forward with timestep dt.
       Also handle tumbling and collisions"""
        following_before = self.is_following_wall() # Check if RTP is following wall a priori
        if (self.time + dt) > self.runtime:
            self.tumble() # If particles exponential runtime is reached, tumble.

        self.set_r_vec(self.get_r() + dt * self.get_v_vec()) # Set new position

        if following_before and not self.is_following_wall(): # Reset moving angle to match
                                                              # intrinsic angle when free from wall
            self.moving_angle = self.free_moving_angle()
            self.hovering_wall = None

        hitwall = self.hits_wall() # check if rtp hits any walls (EXPENSIVE STEP)
        if hitwall:
            for w in hitwall:
                self.collide(w) # Handle collisons for any wall which the rtp collides with

        self.time = self.time + dt # Increment internal clock

    def tumble(self):
        """Handle tumbling process"""
        new = 2 * pi * rand.random() # Pull random angle from uniform [0, 2pi]
        self.angle = new # assign new angle
        self.runtime = rand.expovariate(self.RATE) # assign new runtime pulled from exp. dist.
        self.time = 0 # reset internal clock
        ## Break free from wall
        self.v = self.VELOCITY # set velocity to intrinsic velocity
        self.moving_angle = self.free_moving_angle() # set angle to free angle

    def hits_wall(self):
        """return any wall with which the particle overlaps"""
        walls = []
        # check hovering wall (overzealous?)
        if self.hovering_wall and self.hovering_wall.distance_to(self) < self.RADIUS:
            walls.append(self.hovering_wall)
        # check walls on path
        for w in self.walls_on_path:
            if w.distance_to(self) < self.RADIUS:
                walls.append(w)
        if len(walls) > 0:
            return walls
        else:
            return None

    def get_walls_on_path(self):
        """return walls on the path of RTP"""
        xf = self.x + 1.5 * self.runtime * self.get_v_vec()[0] # 1.5*runtime to account for 
        yf = self.y + 1.5 * self.runtime * self.get_v_vec()[1] # almost-parallel paths with walls
        r0, rf = np.array((self.x, self.y)), np.array((xf, yf)) # Initial and final pos of path
        walls = []
        for w in self.cell.get_walls():
            c1 = cross((w.rf - w.r0), (r0 - w.r0)) # if cross products have different signs,
            c2 = cross((w.rf - w.r0), (rf - w.r0)) # r0 and rf are on opposite sides of wall
            if c1 * c2 < 0:
                walls.append(w)
        return walls

    def new_pos(self, wall, tol=0.005):
        """sets new position relative to colliding wall. tol expresses some margin to avoid
        unnecessary repeated collisions"""
        r0 = wall.r0
        p = self.get_r()
        x = dot(wall.vector, p-r0)/wall.length**2 * wall.vector
        d_vec = ((p - r0) - x) / norm((p - r0) - x)
        self.set_r_vec(self.get_r() + (self.RADIUS - wall.distance_to(self) + tol) * d_vec)

    def collision_angle(self, wall):
        """return the angle of collision with a wall"""
        v = np.array((cos(self.moving_angle), sin(self.moving_angle)))
        w = np.array((cos(wall.angle), sin(wall.angle)))
        c = acos(dot(v, w))
        if c > pi / 2:
            return pi - c
        else:
            return c

    def collide(self, wall):
        """handle collision with a wall"""
        c = self.collision_angle(wall)
        self.v = self.v * cos(c) # velocity reduced to component along wall

        self.new_pos(wall) # set new position: shift away from wall

        ## Figure out which angle to assign to particle
        w = np.array((cos(wall.angle), sin(wall.angle)))
        v = np.array((cos(self.moving_angle), sin(self.moving_angle)))
        sign = dot(v, w)
        if sign >= 0:
            self.moving_angle = wall.angle
        elif sign < 0:
            self.moving_angle = pi + wall.angle

        self.hovering_wall = wall # RTP is now following the wall

class Ball(Particle):
    """A class for a passive 'ping-pong' ball which moves according to collisions with
    RTP particles."""
    pass
