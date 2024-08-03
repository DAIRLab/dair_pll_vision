"""Inertia computations for Franka end effector, given each link's mass and
geometric properties.  Assumes each link's mass is evenly distributed."""

base = 'base'
link = 'link'
tip = 'tip'

masses =  {base: 0.0779312, link: 0.1340688, tip: 0.057}
radii =   {base: 0.0315,    link: 0.0127,    tip: 0.0195}
heights = {base: 0.0096,    link: 0.1016}

def compute_cylinder_inertias(mass, rad, height):
    Ixx = (1/12) * mass * (3*rad**2 + height**2)
    Izz = (1/2) * mass * rad**2

    print(f'Ixx = Iyy: {Ixx} \nIzz: {Izz}\n')

def compute_sphere_inertias(mass, rad):
    Ixx = (2/5) * mass * rad**2
    print(f'Ixx: {Ixx}\n')


print(f'Base:')
compute_cylinder_inertias(masses[base], radii[base], heights[base])

print(f'Link:')
compute_cylinder_inertias(masses[link], radii[link], heights[link])

print(f'Tip:')
compute_sphere_inertias(masses[tip], radii[tip])
