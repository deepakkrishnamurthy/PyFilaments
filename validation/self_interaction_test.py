# Self-interaction module tests
import numpy as np
# Define two line segments in 3D space
r_i = np.array([0,0,0])
r_j = np.array([0,1,0])

# Line segment orientations
dr_hat_i = np.array([1,1,1])
dr_hat_j = np.array([1,1,1])

dr_hat_i = dr_hat_i/(np.dot(dr_hat_i, dr_hat_i)**(1/2))
dr_hat_j = dr_hat_j/(np.dot(dr_hat_j, dr_hat_j)**(1/2))

l_i = 1	# Line lengths
l_j = 2
radius = 1 # Radius of the cylinder whose center-line is give by the line segments.


# calculate the minumum separation between the line segments
r_ji = r_i - r_j

dri_dot_drj = 0
dri_dot_r = 0
drj_dot_r = 0
for index in range(3):
	dri_dot_drj += dr_hat_i[index]*dr_hat_j[index]
	dri_dot_r += dr_hat_i[index]*(r_i[index] - r_j[index])
	drj_dot_r += dr_hat_j[index]*(r_i[index] - r_j[index])



if(np.isclose(dri_dot_drj,1.0, 1e-6)):
	"""
	 Parallel or almost parallel case
	
	"""
	print(50*'*')
	print('Lines are parallel/almost parallel!')
	print(50*'*')


	h_i = 0
	h_j = 0

	# Calculate distance between the parallel lines
	perp_line = r_ji - dri_dot_r*dr_hat_i

	d_perp2 = 0.0
	for index in range(3):
		d_perp2 += perp_line[index]*perp_line[index]

	d_perp = d_perp2**(1/2)
	print(50*'*')
	print('Perpendicular distance between the line segments (parallel case): {}'.format(d_perp))
	print(50*'*')
	# Calculate the line of closest approach in the plane containing the two lines.

	# Let gamma and delta be the parameter along each line. We want to find gamma and delta that gives the closest line
	# We can show that the distance is minimized when gamma- delta = -r_ji.dr_hat_i = -r_ji.dr_hat_j 
	# Now we choose the gamma_min, gamma_max and delta_min, delta_max that satisfy this constraint.

	#Find gamma_min
	if(-dri_dot_r<0):
		gamma_min = 0.0
	elif (-dri_dot_r >= 0 and -dri_dot_r < l_i):
		gamma_min = -dri_dot_r
	else:
		gamma_min = l_i

	# Find delta_min
	if(dri_dot_r < 0):
		delta_min = 0
	elif(dri_dot_r >=0 and dri_dot_r < l_j):
		delta_min = dri_dot_r
	else:
		delta_min = l_j

	# Find gamma_max
	if(l_j - dri_dot_r < 0):
		gamma_max = 0.0
	elif(l_j - dri_dot_r >=0 and l_j - dri_dot_r < l_i):
		gamma_max = l_j - dri_dot_r
	else:
		gamma_max = l_i

	# Find delta_max:
	if(l_i + dri_dot_r < 0):
		delta_max = 0.0
	elif(l_i + dri_dot_r >= 0 and l_i + dri_dot_r < l_j):
		delta_max = l_i + dri_dot_r 
	else:
		delta_max  = l_j

	print(50*'*')
	print("gamma_min, gamma_max : ({}, {})".format(gamma_min, gamma_max))
	print("delta_min, delta_max : ({}, {})".format(delta_min, delta_max))
	print(50*'*')
	# Find gamma_final and delta_final by choosing the mid-point of the overlap if there is overlap
	if (gamma_min!=gamma_max and delta_min!= delta_max):
		# In case the two lines have an overlap region
		gamma_min_final = (gamma_min + gamma_max)/2.0
		delta_min_final = (delta_min + delta_max)/2.0

		print(50*'*')
		print('Parallel lines with overlap region')
		print('Closest line is the same as the perpendicular line')
		print("Overlapping parallel lines \n (gamma, delta)= ({}, {})".format(gamma_min_final, delta_min_final))
		print(50*'*')
		d_par = 0
	else:
		gamma_min_final = gamma_max
		delta_min_final = delta_max
		print(50*'*')
		print('Parallel lines without overlap region')
		print('Closest line is the same as the perpendicular line')
		print("Non-overlapping parallel lines \n (gamma, delta)= ({}, {})".format(gamma_min_final, delta_min_final))
		print(50*'*')
		d_par = (np.dot(r_ji, r_ji) + (gamma_min_final - delta_min_final)**2 + 2*dri_dot_r*(gamma_min_final - delta_min_final))**(1/2)

		print(50*'*')
		print('Actual closest distance that lies on both lines (parallel case): {}'.format(d_par))
		print(50*'*')

else:
	"""
		If the lines are not parallel then find the unique common perpendicular line. 
	"""
	print(50*'*')
	print('Lines are not parallel... calculating common normal')
	print(50*'*')

	h_i = (drj_dot_r*dri_dot_drj - dri_dot_r)/((1 - dri_dot_drj*dri_dot_drj))
	h_j = (drj_dot_r - dri_dot_drj*dri_dot_r)/((1 - dri_dot_drj*dri_dot_drj))

	d_min_ji = r_ji +h_i*dr_hat_i - h_j*dr_hat_j
	d_perp = np.dot(d_min_ji, d_min_ji)**(1/2)
	print(50*'*')
	print('Perpendicular distance between the line segments (non-parallel case): {}'.format(d_perp))
	print(50*'*')
	# Now find the minimum in-plane distance (for non-parallel lines)
	# Adapted from Allen et al. Adv. Chem. Phys. Vol LXXXVI, 1003, p.1.

	# If the intersection point in-plane lies within the line-segments, then the inplane distance = 0

	if(h_i >= 0 and h_i <= l_i and h_j >= 0 and h_j <= l_j):
		gamma_min_final, delta_min_final = 0,0
		d_par = 0 
		# Origin lies in the range
		print('Origin lies in the range')

	else:

		gamma_1 = -h_i 
		gamma_2 = -h_i + l_i
		gamma_m = gamma_1
		# Choose the line closest to the origin
		if(gamma_1*gamma_1 > gamma_2*gamma_2):
			gamma_m = gamma_2

		delta_1 = -h_j
		delta_2 = -h_j +l_j
		delta_m = delta_1

		if(delta_1*delta_1 > delta_2*delta_2):
			delta_m = delta_2

		# Optimize delta on gamma_m
		gamma_min = gamma_m
		delta_min = gamma_m*dri_dot_drj

		if(delta_min + h_j >= 0 and delta_min + h_j <= l_j):
			delta_min = delta_min
		else:
			delta_min = delta_1
			a1 = delta_min - delta_1
			a2 = delta_min - delta_2
			if(a1*a1 > a2*a2):
				delta_min = delta_2

		# Distance at this gamma and delta value
		f1 = gamma_min**2 + delta_min**2 - 2*gamma_min*delta_min*dri_dot_drj
		gamma_min_final = gamma_min
		delta_min_final = delta_min

		# Now choose the line delta_m and optimize gamma
		delta_min = delta_m
		gamma_min = delta_m*dri_dot_drj

		if(gamma_min + h_i >= 0 and gamma_min + h_i <= l_i):
			gamma_min = gamma_min
		else:
			gamma_min = gamma_1
			b1 = gamma_min - gamma_1
			b2 = gamma_min - gamma_2
			if(b1*b1 > b2*b2):
				gamma_min = gamma_2

		f2 = gamma_min**2 + delta_min**2 -2*gamma_min*delta_min*dri_dot_drj

		if(f1 < f2):
			pass
		else:
			delta_min_final = delta_min
			gamma_min_final = gamma_min

		print(f1, f2)
		d_par = (min(f1,f2))**(1/2)
		print(50*'*')
		print('In plane minimum distance between the line segments (non-parallel case): {}'.format(d_par))
		print(50*'*')

print(50*'*')
print('Parameters for closest approach between the two lines (perp): \n')
print("h_i : {}".format(round(h_i, 3)))
print("h_j : {}".format(round(h_j, 3)))
print(50*'*')

print(50*'*')
print('Parameters for closest approach between the two lines (parallel): \n')
print("gamma_min_final : {}".format(round(gamma_min_final, 3)))
print("delta_min_final : {}".format(round(delta_min_final, 3)))
print(50*'*')


d_total = (d_perp**2 + d_par**2)**(1/2)

line_closest_approach = r_ji + (h_i + gamma_min_final)*dr_hat_i - (h_j + delta_min_final)*dr_hat_j

dmin_mag = (np.dot(line_closest_approach, line_closest_approach)**(1/2))

print(50*'*')
print('Distance of closest approach between the line segments (perp + parallel): {}'.format(round(d_total, 3)))
print('Distance of closest approach between the line segments (total): {}'.format(round(dmin_mag, 3)))
print('Closest point on line i: {}'.format(r_i + (h_i + gamma_min_final)*dr_hat_i))
print('Closest point on line j: {}'.format(r_j + (h_j + delta_min_final)*dr_hat_j))
print(50*'*')


# Separation forces on the line-segments
ljrmin = 2.0*radius
ljeps = 0.1

dmin_x = line_closest_approach[0]
dmin_y = line_closest_approach[1]
dmin_z = line_closest_approach[2]

idr     = 1.0/dmin_mag
rminbyr = ljrmin*idr 
fac   = ljeps*(np.power(rminbyr, 12) - np.power(rminbyr, 6))*idr*idr
# if(epsilon>0):
f_x = fac*dmin_x
f_y = fac*dmin_y
f_z = fac*dmin_z

torque_scale_factor = (h_i + gamma_min_final)/l_i

print(torque_scale_factor)

F_sc_i = np.zeros(3)
F_sc_i1 = np.zeros(3)

F_sc_i[0] = f_x*(1 - torque_scale_factor)
F_sc_i[1] = f_y*(1 - torque_scale_factor)
F_sc_i[2] = f_z*(1 - torque_scale_factor)

F_sc_i1[0] = f_x*torque_scale_factor
F_sc_i1[1] = f_y*torque_scale_factor
F_sc_i1[2] = f_z*torque_scale_factor

print('Force on i: {}'.format(np.dot(F_sc_i, F_sc_i)**(1/2)))
print('Force on i+1: {}'.format(np.dot(F_sc_i1, F_sc_i1)))

# Visualize the results


r_i_0 = r_i
r_i_1 = r_i + dr_hat_i*l_i

r_j_0 = r_j
r_j_1 = r_j + dr_hat_j*l_j

print('Line i extent: {} to {}'.format(r_i_0, r_i_1))
print('Line j extent: {} to {}'.format(r_j_0, r_j_1))


line_closest_approach_0 = r_i + (h_i + gamma_min_final)*dr_hat_i
line_closest_approach_1 = r_j + (h_j + delta_min_final)*dr_hat_j

print(h_i)
print(h_j)
print(line_closest_approach_0)
print(line_closest_approach_1)

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot([r_i_0[0], r_i_1[0]], [r_i_0[1], r_i_1[1]], [r_i_0[2], r_i_1[2]], label='Line i', color = 'b')
ax.plot([r_j_0[0], r_j_1[0]], [r_j_0[1], r_j_1[1]], [r_j_0[2], r_j_1[2]], label='Line j', color = 'g')
ax.plot([line_closest_approach_0[0], line_closest_approach_1[0]], [line_closest_approach_0[1], line_closest_approach_1[1]], [line_closest_approach_0[2], line_closest_approach_1[2]], label='Line of shortest approach', color = 'r')
ax.scatter(line_closest_approach_0[0],line_closest_approach_0[1],line_closest_approach_0[2], marker = 'o')
ax.scatter(line_closest_approach_1[0],line_closest_approach_1[1],line_closest_approach_1[2], marker = 's')
# ax.quiver([r_i_0[0], r_i_1[0]], [r_i_0[1], r_i_1[1]], [r_i_0[2], r_i_1[2]], [F_sc_i[0], F_sc_i1[0]], [F_sc_i[1], F_sc_i1[1]], [F_sc_i[2], F_sc_i1[2]], length = 0.5, normalize = True,color = 'm', label = 'Force on i')

ax.legend()

plt.show()


fig = plt.figure()
ax = fig.gca()
ax.plot([r_i_0[0], r_i_1[0]], [r_i_0[1], r_i_1[1]], label='Line i', color = 'b')
ax.plot([r_j_0[0], r_j_1[0]], [r_j_0[1], r_j_1[1]], label='Line j', color = 'g')
ax.plot([line_closest_approach_0[0], line_closest_approach_1[0]], [line_closest_approach_0[1], line_closest_approach_1[1]], label='Line of shortest approach', color = 'r')
ax.scatter(line_closest_approach_0[0],line_closest_approach_0[1], marker = 'o')
ax.scatter(line_closest_approach_1[0],line_closest_approach_1[1], marker = 's')
# ax.quiver([r_i_0[0], r_i_1[0]], [r_i_0[1], r_i_1[1]], [r_i_0[2], r_i_1[2]], [F_sc_i[0], F_sc_i1[0]], [F_sc_i[1], F_sc_i1[1]], [F_sc_i[2], F_sc_i1[2]], length = 0.5, normalize = True,color = 'm', label = 'Force on i')
ax.legend()
ax.set_aspect('equal')

plt.show()