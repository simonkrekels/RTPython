import matplotlib.pyplot as plt
from math import asin, acos, atan2, sin, cos, sqrt, pi
import numpy as np
import random as rand
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

def norm(array):
    return (array[0]**2 + array[1]**2)**0.5

def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

class Wall():

    THICKNESS = 1.5

    def __init__(self, x0, y0, xf, yf):
        self.r0 = np.array((x0, y0))
        self.rf = np.array((xf, yf))
        self.angle = self.wall_angle()
        if self.angle in [0, pi]:
            self.horizontal = True
            self.vertical = False
        elif self.angle in [pi/2, 3*pi/2]:
            self.horizontal = False
            self.vertical = True
        else:
            self.horizontal = False
            self.vertical = False
        self.length = self.wall_length()
        self.vector = self.wall_vector()
        self._cell = None

    def __str__(self):
        return f"Wall: {self.r0} -- {self.rf}"

    @property
    def cell(self):
        return self._cell

    @cell.setter
    def cell(self, c):
        self._cell = c

    def distance_to(self, particle):
        if self.horizontal:
            comp_x = [self.r0[0] - particle.x, self.rf[0] - particle.x]
            if all(i >= 0 for i in comp_x): # particle to the left of wall
                return sqrt(min(comp_x)**2 + (particle.y - self.r0[1])**2)
            elif all(i <= 0 for i in comp_x): # particle to the right of wall
                return sqrt(max(comp_x)**2 + (particle.y - self.r0[1])**2)
            else:
                return abs(particle.y - self.r0[1])
        elif self.vertical:
            comp_y = [self.r0[1] - particle.y, self.rf[1] - particle.y]
            if all(j <= 0 for j in comp_y): # particle above wall 
                return sqrt(max(comp_y)**2 + (particle.x - self.r0[0])**2)
            elif all(j >= 0 for j in comp_y): # particle below wall
                return sqrt(min(comp_y)**2 + (particle.x - self.r0[0])**2)
            else:
                return abs(particle.x - self.r0[0])
        else:
            rp = particle.get_r()
            if all(self.r0 == rp) or all(self.rf == rp):
                return 0
            if acos(dot((rp - self.r0) / norm(rp - self.r0),
                        (self.rf - self.r0) / norm(self.rf - self.r0))) > pi / 2:
                return norm(rp - self.r0)
            if acos(dot((rp - self.rf) / norm(rp - self.rf),
                        (self.r0 - self.rf) / norm(self.r0 - self.rf))) > pi / 2:
                return norm(rp - self.rf)
            else:
                return abs(cross(self.r0 - self.rf, self.r0 - rp)) / norm(self.rf - self.r0)

    def wall_length(self):
        return norm(self.rf - self.r0)

    def wall_vector(self):
        return self.rf - self.r0

    def wall_angle(self):
        line = self.rf - self.r0
        t = acos((line).dot(np.array((1,0))) / norm(line))
        s = asin(cross(line, np.array((0,1))) / norm(line))
        if s <= 0:
            return t
        else:
            return -t

    def update(self):
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
        if pos not in ['r0', 'rf', 'mid']:
            raise ValueError(f"Wall cannot rotate around {pos}")
        l = self.wall_length()
        t = self.wall_angle()
        if pos == 'r0':
            rfnew = self.r0 + np.array((l * cos(t - angle), l * sin(t - angle)))
            if rfnew[0] <= self.cell.cell_size[0] and rfnew[1] <= self.cell.cell_size[1]:
                self.rf = rfnew
        elif pos == 'rf':
            r0new = self.rf - np.array((l * cos(t - angle), l * sin(t - angle)))
            if r0new[0] <= self.cell.cell_size[0] and r0new[1] <= self.cell.cell_size[1]:
                self.r0 = r0new
        elif pos == 'mid':
            rmid = (self.r0 + self.rf)/2
            rfnew = rmid + np.array((l * cos(t - angle), l * sin(t - angle)))
            r0new = rmid - np.array((l * cos(t - angle), l * sin(t - angle)))
            if (rfnew[0] <= self.cell.cell_size[0] and rfnew[1] <= self.cell.cell_size[1] and
                    r0new[0] <= self.cell.cell_size[0] and r0new[1] <= self.cell.cell_size[1]):
                self.r0 = r0new
                self.rf = rfnew
        self.update()

    def translate(self, dx, dy):
        dr = np.array((dx, dy))
        r0new = self.r0 + dr
        rfnew = self.rf + dr
        if (rfnew[0] <= self.cell.cell_size[0] and rfnew[1] <= self.cell.cell_size[1] and
                    r0new[0] <= self.cell.cell_size[0] and r0new[1] <= self.cell.cell_size[1]):
            self.r0 = r0new
            self.rf = rfnew
        self.update()

    def draw(self, ax):
        # ax.plot([self.r0[0], self.rf[0]], [self.r0[1], self.rf[1]], color='black', linewidth=self.THICKNESS)
        line = Line2D([self.r0[0], self.rf[0]], [self.r0[1], self.rf[1]], color='black', linewidth=self.THICKNESS)
        ax.add_line(line)
        return line

class Cell():

    walls = []
    particles = []
    velocity = 0.5
    decay_rate = 0.5
    particle_radius = 0.15
    field = 0
    pbc = False

    def __init__(self, xsize=10, ysize=10, particles=[], default_walls=True):
        self.cell_size = np.array((xsize, ysize))
        particles = []
        self.add_wall(Wall(0, ysize, xsize, ysize))
        self.add_wall(Wall(0, 0, xsize, 0))
        if default_walls:
            self.add_wall(Wall(0, 0, 0, ysize/2))
            self.add_wall(Wall(xsize, 0, xsize, ysize/2))
            self.add_wall(Wall(xsize, ysize/2, xsize/2, ysize/2))
        for p in particles:
            self.add_particle(p)
        self._left_flux = 0
        self._right_flux = 0

    @property
    def left_flux(self):
        return self._left_flux

    @left_flux.setter
    def left_flux(self, value):
        self._left_flux = value

    @property
    def right_flux(self):
        return self._right_flux

    @right_flux.setter
    def right_flux(self, value):
        self._right_flux = value

    def update_cells(self):
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

    def set_particle_radius(self, radius):
        self.particle_radius = radius
        self.update_cells()

    def get_particle_radius(self):
        return self.particle_radius

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

    VELOCITY = 0.5
    RADIUS = 0.1
    RATE = 5
    FIELD = 0
    walls_on_path = []

    def __init__(self, x_pos=0, y_pos=0, angle=0, cell=None, pbc=False):
        self._x = x_pos
        self._y = y_pos
        self._angle = angle
        self._time = 0
        self._runtime = rand.expovariate(self.RATE)
        self._cell = cell
        self._v = self.VELOCITY
        self.moving_angle = self.free_moving_angle()
        self.PERIODIC_BC = pbc
        self._hovering_wall = None

    @property
    def x(self):
        """The x-coordinate"""
        return self._x

    @x.setter
    def x(self, value):
        if self.PERIODIC_BC:
            if value > self.cell.cell_size[0]:
                self.cell.right_flux += 1
            elif value < 0:
                self.cell.left_flux += 1
            self._x = value % self.cell.cell_size[0]
        else:
            self._x = value

    @property
    def y(self):
        """The y-coordinate"""
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    def get_r(self):
        return np.array((self.x, self.y))

    def set_r(self, x, y):
        self.x = x
        self.y = y

    def set_r_vec(self, r):
        if len(r) != 2:
            raise TypeError
        self.set_r(r[0], r[1])

    @property
    def angle(self):
        """The particle's intrinsic angle"""
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value

    @property
    def time(self):
        """The particle's internal clock"""
        return self._time

    @time.setter
    def time(self, value):
        self._time = value

    @property
    def runtime(self):
        """The particle's lifetime. When Particle.time exceeds Particle.runtime,
        the particle tumbles"""
        return self._runtime

    @runtime.setter
    def runtime(self, value):
        self._runtime = value

    @property
    def cell(self):
        """The cell this particle exists in"""
        return self._cell

    @cell.setter
    def cell(self, obj):
        self.VELOCITY = obj.get_velocity()
        self.RADIUS = obj.get_particle_radius()
        self.RATE = obj.get_decay_rate()
        self.FIELD = obj.get_field()
        self.PERIODIC_BC = obj.get_pbc()
        self._cell = obj

    @cell.deleter
    def cell(self):
        return self._cell

    def update_cell(self):
        if self.cell is not None:
            self.VELOCITY = self.cell.get_velocity()
            self.RADIUS = self.cell.get_particle_radius()
            self.RATE = self.cell.get_decay_rate()
            self.FIELD = self.cell.get_field()
            self.PERIODIC_BC = self.cell.get_pbc()

    @property
    def v(self):
        """The particle's current velocity"""
        return self._v

    @v.setter
    def v(self, value):
        self._v = value

    def get_v_vec(self):
        if self.hovering_wall:
            return np.array((self.v * cos(self.moving_angle) + self.FIELD, self.v * sin(self.moving_angle)))
        else:
            v_f = sqrt((self.v * cos(self.angle) + self.FIELD)**2 + (self.v * sin(self.angle))**2)
            return np.array((v_f * cos(self.moving_angle), v_f * sin(self.moving_angle)))

    @property
    def moving_angle(self):
        """The angle the particle is moving in"""
        return self._moving_angle

    @moving_angle.setter
    def moving_angle(self, value):
        self._moving_angle = value
        if self.cell:
            self.walls_on_path = self.get_walls_on_path()

    def free_moving_angle(self):
        return atan2(self.VELOCITY * sin(self.angle), (self.VELOCITY * cos(self.angle) + self.FIELD))

    def is_following_wall(self):
        if self.hovering_wall and self.hovering_wall.distance_to(self) < 3 * self.RADIUS:
            return True
        else:
            return False

    @property
    def hovering_wall(self):
        return self._hovering_wall

    @hovering_wall.setter
    def hovering_wall(self, wall):
        self._hovering_wall = wall

    def run(self, dt=1):
        following_before = self.is_following_wall()
        # pos_before = self.get_r()
        # v_vec_before = self.get_v_vec()
        # angle_before = self.angle
        # moving_angle_before = self.moving_angle
        # walls_on_path_before = self.walls_on_path
        if (self.time + dt) > self.runtime:
            self.tumble()

        if following_before and not self.is_following_wall():
            self.moving_angle = self.free_moving_angle()
            self.hovering_wall = None

        self.set_r_vec(self.get_r() + dt * self.get_v_vec())

        hitwall = self.hits_wall()
        if hitwall:
            for w in hitwall:
                self.collide(w)

        self.time = self.time + dt

        # print(f"Run log:\npos: {pos_before} to {self.get_r()}\nv_vec: {v_vec_before} to {self.get_v_vec()}\nangle: {angle_before} to {self.angle}\nmoving angle: {moving_angle_before} to {self.moving_angle}\n walls on path: {walls_on_path_before} to {self.walls_on_path}")

    def tumble(self):
        new = 2 * pi * rand.random()
        self.angle = new
        self.runtime = rand.expovariate(self.RATE)
        self.time = 0
        self.v = self.VELOCITY
        self.moving_angle = self.free_moving_angle()

    def hits_wall(self):
        walls = []
        if self.hovering_wall and self.hovering_wall.distance_to(self) < self.RADIUS:
            walls.append(self.hovering_wall)
        for w in self.walls_on_path:
            if w.distance_to(self) < self.RADIUS:
                walls.append(w)
        if len(walls) > 0:
            return walls
        else:
            return None

    def get_walls_on_path(self):
        xf = self.x + 1.5 * self.runtime * self.get_v_vec()[0]
        yf = self.y + 1.5 * self.runtime * self.get_v_vec()[1]
        r0, rf = np.array((self.x, self.y)), np.array((xf, yf))
        walls = []
        for w in self.cell.get_walls():
            c1 = cross((w.rf - w.r0), (r0 - w.r0))
            c2 = cross((w.rf - w.r0), (rf - w.r0))
            if c1 * c2 < 0:
                walls.append(w)
        return walls

    def new_pos(self, wall, tol=0.005):
        r0 = wall.r0
        p = self.get_r()
        x = wall.vector.dot(p-r0)/wall.length**2 * wall.vector
        d_vec = ((p - r0) - x) / norm((p - r0) - x)
        self.set_r_vec(self.get_r() + (self.RADIUS - wall.distance_to(self) + tol) * d_vec)

    def collision_angle(self, wall):
        v = np.array((cos(self.moving_angle), sin(self.moving_angle)))
        w = np.array((cos(wall.angle), sin(wall.angle)))
        c = acos(v.dot(w))
        if c > pi / 2:
            return pi - c
        else:
            return c

    def collide(self, wall):
        c = self.collision_angle(wall)
        self.v = self.v * cos(c)
        originalpos = self.get_r()
        originalangle = self.moving_angle

        self.new_pos(wall)

        w = np.array((cos(wall.angle), sin(wall.angle)))
        v = np.array((cos(self.moving_angle), sin(self.moving_angle)))
        sign = v.dot(w)
        if sign >= 0:
            self.moving_angle = wall.angle
        elif sign < 0:
            self.moving_angle = pi + wall.angle
        self.hovering_wall = wall
        # print(f"collision! Set pos from {originalpos} to {self.get_r()} and angle from {originalangle} to {self.moving_angle}")

    def draw(self, ax):
        circle = Circle(xy=self.get_r(), radius=self.RADIUS)
        ax.add_patch(circle)
        return circle


