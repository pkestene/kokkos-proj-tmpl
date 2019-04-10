import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex=True)

size=np.array([32,48,64,92,128,160,192,224,256,320,384,448,512,640,768])

# test_stencil_3d_flat
v0=np.array([52.6712,109.303,135.742,152.486,158.096,157.501,157.385,160.092,156.636,161.104,161.877,162.767,160.931,161.816,161.765])

# test_stencil_3d_flat_1d_array
v1=np.array([74.9241,250.733,448.209,783.461,1016.89,795.825,692.858,761.999,705.385,717.463,685.021,707.628,671.067,684.678,681.514])

# test_stencil_3d_flat_vector without views
v2=np.array([72.3363,155.918,214.539,252.449,277.179,281.263,283.796,281.426,241.291,293.143,262.558,290.445,277.061,286.434,283.173])

# test_stencil_3d_flat_vector with    views
v3=np.array([78.818,232.655,417.371,705.697,961.322,793.413,707.83,734.292,713.362,717.755,684.552,716.978,680.186,689.099,681.567])

# test_stencil_3d_range
v4=np.array([44.6049,75.2708,106.041,106.898,120.309,122.601,126.632,128.483,124.558,128.892,130.49,131.844,123.767,130.991,128.248])

# test_stencil_3d_range_vector
v5=np.array([40.0186,109.545,194.712,381.576,631.251,562.525,630.712,773.804,713.362,532.98,587.399,668.471,679.378,640.249,678.248])

# test_stencil_3d_range_hierarchical
v6=np.array([62.0317,188.368,339.617,655.67,872.52,574.878,656.107,768.521,699.969,649.235,645.645,666.89,666.085,685.149,678.747])

# test_stencil_3d_range_hierarchical_linearized
v7=np.array([67.6623,192.272,346.644,662.738,969.6,578.386,658.975,786.471,716.107,667.372,695.14,682.161,691.577,689.757,682.926])

plt.plot(size,v0, label='# test_stencil_3d_flat')
plt.plot(size,v1, label='# test_stencil_3d_flat_1d_array')
plt.plot(size,v2, label='# test_stencil_3d_flat_vector without views')
plt.plot(size,v3, label='# test_stencil_3d_flat_vector with    views')
plt.plot(size,v4, label='# test_stencil_3d_range')
plt.plot(size,v5, label='# test_stencil_3d_range_vector')
plt.plot(size,v6, label='# test_stencil_3d_range_hierarchical')
plt.plot(size,v7, label='# test_stencil_3d_range_hierarchical_linearized')
plt.grid(True)
plt.title('3d 7 points stencil kernel performance')
plt.xlabel('N - linear size')
plt.ylabel(r'Bandwidth (GBytes/s)')
plt.legend()
plt.show()
