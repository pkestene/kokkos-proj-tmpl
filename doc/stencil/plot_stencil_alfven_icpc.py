import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex=True)

size=np.array([32,48,64,92,128,160,192,224,256,320,384,448,512,640,768])

# test_stencil_3d_flat
v0=np.array([48.8491,73.3612,78.3267,84.6014,86.4136,67.6814,87.9137,87.9332,86.9441,87.9198,85.6755,81.9944,81.9694,86.6875,86.197])

# test_stencil_3d_flat_1d_array
v1=np.array([122.508,333.81,645.19,1260.72,1334,1786.66,1681.48,1105.31,589.775,366.811,361.997,321.855,309.337,304.52,318.488])

# test_stencil_3d_flat_vector without views
v2=np.array([89.1496,207.31,405.974,436.93,573.736,614.207,590.538,541.034,511.78,408.489,362.762,317.536,305.407,302.745,317.247])

# test_stencil_3d_flat_vector with    views
v3=np.array([114.038,296.473,731.484,1356.73,1655.74,1767.89,1783.44,1123.59,591.137,420.242,364.339,322.389,307.318,300.421,318.317])

# test_stencil_3d_range
v4=np.array([93.0475,157.518,275.021,322.905,321.099,341.838,343.502,323.243,307.539,302.648,310.719,321.586,316.129,303.471,303.024])

# test_stencil_3d_range_vector
v5=np.array([123.889,281.659,416.055,883.713,1373.27,1978.01,1215.95,987.193,482.77,420.232,349.915,309.43,291.577,286.587,310.429])

# test_stencil_3d_range_vector2
v6=np.array([170.909,350.356,712.715,1058.55,1315.1,1865.27,1556.32,988.336,543.209,416.135,348.526,307.004,302.514,299.531,309.789])

plt.plot(size,v0, label='# test_stencil_3d_flat')
plt.plot(size,v1, label='# test_stencil_3d_flat_1d_array')
plt.plot(size,v2, label='# test_stencil_3d_flat_vector without views')
plt.plot(size,v3, label='# test_stencil_3d_flat_vector with    views')
plt.plot(size,v4, label='# test_stencil_3d_range')
plt.plot(size,v5, label='# test_stencil_3d_range_vector')
plt.plot(size,v6, label='# test_stencil_3d_hierarchical')
plt.grid(True)
plt.title('3d 7 points stencil kernel performance')
plt.xlabel('N - linear size')
plt.ylabel(r'Bandwidth (GBytes/s)')
plt.legend()
plt.show()
