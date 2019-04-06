import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex=True)

size=np.array([32,48,64,92,128,160,192,224,256,320,384,448,512,640,768])

# test_stencil_3d_flat
v0=np.array([58.6609,197.804,469.897,594.793,619.813,626.039,631.271,633.735,634.21,634.137,667.889,699.473,707.576,706.431,703.223])

# test_stencil_3d_flat_1d_array
v1=np.array([58.5091,198.505,192.456,343.894,598.686,882.378,1108.69,1205.08,1203.21,1000.15,921.188,927.959,984.483,879.279,896.083])

# test_stencil_3d_flat_vector without views
v2=np.array([39.1692,132.947,211.245,403.476,752.888,986.079,1052.96,1131.49,1016.65,888.436,859.658,835.339,814.432,748.524,737.256])

# test_stencil_3d_flat_vector with    views
v3=np.array([59.1373,197.477,396.377,400.179,698.131,1016.1,1134.69,1173.63,952.103,963.566,1032.34,1056.67,1104.72,1002.01,1025.61])

# test_stencil_3d_range
v4=np.array([108.261,363.849,588.36,634.395,703.536,716.178,721.067,723.758,725.216,726.56,726.96,727.157,727.221,727.128,727.075])

# test_stencil_3d_range_vector
v5=np.array([64.05,155.277,141.388,270.251,508.579,697.031,900.695,1080.9,827.759,1058.79,994.15,954.766,1054.37,948.56,993.995])

# test_stencil_3d_range_hierarchical
v6=np.array([53.606,143.143,129.967,199.442,275.552,338.721,392.875,446.443,493.481,578.535,661.547,709.45,713.453,729.744,580.055])

plt.plot(size,v0, label='# test_stencil_3d_flat')
plt.plot(size,v1, label='# test_stencil_3d_flat_1d_array')
plt.plot(size,v2, label='# test_stencil_3d_flat_vector without views')
plt.plot(size,v3, label='# test_stencil_3d_flat_vector with    views')
plt.plot(size,v4, label='# test_stencil_3d_range')
plt.plot(size,v5, label='# test_stencil_3d_range_vector')
plt.plot(size,v6, label='# test_stencil_3d_range_hierarchical')
plt.grid(True)
plt.title('3d 7 points stencil kernel performance')
plt.xlabel('N - linear size')
plt.ylabel(r'Bandwidth (GBytes/s)')
plt.legend()
plt.show()
