import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex=True)

size=np.array([32,48,64,92,128,160,192,224,256,320,384,448,512,640,768])

# test_stencil_3d_flat
v0=np.array([44.2459,105.949,144.139,174.016,184.671,194.501,196.642,196.731,177.89,200.677,192.065,201.532,190.365,197.931,196.185])

# test_stencil_3d_flat_1d_array
v1=np.array([55.1825,190.3,357.202,679.102,758.284,566.934,643.835,718.895,618.745,733.888,682.823,700.978,743.972,662.602,621.365])

# test_stencil_3d_flat_vector without views
v2=np.array([56.0261,132.649,189.775,243.338,261.837,274.96,285.364,285.199,260.528,298.011,292.385,296.686,288.191,299.913,298.481])

# test_stencil_3d_flat_vector with    views
v3=np.array([57.6415,171.799,315.555,637.279,736.653,561.606,631.089,716.88,605.664,711.186,662.889,711.336,716.666,652.734,625.297])

# test_stencil_3d_range
v4=np.array([37.1771,67.2865,102.668,109.29,121.503,125.658,127.861,128.076,116.579,131.064,129.7,132.321,127.452,131.485,130.129])

# test_stencil_3d_range_vector
v5=np.array([25.2037,67.2865,113.498,215.005,358.476,429.597,541.329,672.011,641.502,721.902,714.854,751.644,706.375,596.027,639.056])

# test_stencil_3d_range_hierarchical
v6=np.array([41.9661,119.608,216.786,395.586,584.094,646.315,664.785,742.643,619.511,724.304,721.69,766.685,735.894,591.914,622.288])

# test_stencil_3d_range_hierarchical_linearized
v7=np.array([45.6229,121.568,222.545,425.198,607.282,657.681,677.298,757.675,608.496,722.412,722.465,768.423,730.523,596.624,629.725])

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
