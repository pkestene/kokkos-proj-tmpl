import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex=True)

size=np.array([32,48,64,92,128,160,192,224,256,320,384,448,512,640,768])

# test_stencil_3d_flat
v0=np.array([52.2333,58.7392,58.7557,59.5842,58.8131,58.3216,58.9686,58.1892,57.8252,58.1866,57.6563,56.9009,57.3315,57.0107,57.3183])

# test_stencil_3d_flat_1d_array
v1=np.array([221.843,403.08,681.868,972.447,1121.75,1126.03,686.078,345.504,266.589,239.924,204.088,182.531,172.417,169.897,166.607])

# test_stencil_3d_flat_vector without views
v2=np.array([188.152,326.947,365.078,380.846,378.187,392.872,378.372,317.94,263.673,239.55,204.738,182.507,172.439,169.187,167.108])

# test_stencil_3d_flat_vector with    views
v3=np.array([171.297,399.822,699.143,826.522,1172.69,1171.87,676.803,347.263,267.003,240.688,204.527,181.873,172.771,169.438,166.851])

# test_stencil_3d_range
v4=np.array([135.742,196.862,212.498,221.745,221.151,222.128,217.457,209.212,204.94,205.611,203.355,197.984,197.333,195.121,194.516])

# test_stencil_3d_range_vector
v5=np.array([193.321,447.765,551.695,1096.68,996.019,998.467,681.44,342.828,267.283,242.094,204.549,184.232,172.419,169.977,167.072])

# test_stencil_3d_range_hierarchical
v6=np.array([242.651,455.319,729.21,877.345,1088.04,1112.92,680.976,341.09,266.6,241.525,204.657,182.471,172.556,169.754,167.089])

# test_stencil_3d_range_hierarchical_linearized
v7=np.array([175.396,452.198,709.362,1127.44,1172.69,1189.69,684.891,344.346,258.272,241.405,204.002,181.96,172.564,169.494,166.95])

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
