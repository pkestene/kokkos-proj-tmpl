import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex=True)

size=np.array([32,48,64,92,128,160,192,224,256,320,384,448,512,640,768])

# test_stencil_3d_flat
v0=np.array([58.1922,110.294,127.209,134.969,140.676,141.277,141.374,144.239,157.459,168.624,196.267,209.441,207.175,208.081,208.333])

# test_stencil_3d_flat_1d_array
v1=np.array([41.7267,93.722,136.871,240.022,355.028,351.829,358.337,354.625,353.377,362.729,358.962,360.435,346.413,213.682,100.158])

# test_stencil_3d_flat_vector without views
v2=np.array([40.1798,87.367,133.774,235.09,349.663,344.042,347.52,349.733,344.407,351.272,338.384,334.314,310.884,223.896,103.898])

# test_stencil_3d_flat_vector with    views
v3=np.array([45.469,103.473,156.551,267.055,358.338,348.307,347.745,348.153,330.893,331.867,312.034,307.488,280.879,214.337,96.3619])

# test_stencil_3d_range
v4=np.array([86.4743,138.54,148.47,152.719,166.619,168.032,168.633,168.006,167.285,165.459,166.157,165.54,166.53,166.725,166.859])

# test_stencil_3d_range_vector
v5=np.array([39.1415,91.2124,129.642,137.005,182.047,207.868,201.294,203.938,213.242,203.572,202.346,197.411,198.256,199.84,147.874])

# test_stencil_3d_range_hierarchical
v6=np.array([53.8121,112.303,187.594,237.61,289.938,291.899,272.009,237.575,235.693,210.065,199.708,180.204,179.595,168.836,168.228])

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
