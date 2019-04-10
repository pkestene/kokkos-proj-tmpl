import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex=True)

size=np.array([32,48,64,92,128,160,192,224,256,320,384,448,512,640,768])

# test_stencil_3d_flat
v0=np.array([50.9033,89.7968,99.0272,103.223,107.82,104.447,105.042,105.02,104.649,104.785,105.255,105.388,105.641,104.831,105.794])

# test_stencil_3d_flat_1d_array
v1=np.array([91.8173,272.356,458.13,732.915,1068.62,783.463,670.51,643.159,625.292,680.896,628.625,673.774,637.008,659.332,651.841])

# test_stencil_3d_flat_vector without views
v2=np.array([85.399,166.22,203.143,238.889,259.807,256.249,251.737,254.716,246.316,265.536,257.057,263.083,260.008,262.392,263.242])

# test_stencil_3d_flat_vector with    views
v3=np.array([102.758,297.463,499.778,791.171,1112.99,816.146,665.764,646.3,616.104,683.553,666.055,677.295,641.527,665.313,656.64])

# test_stencil_3d_range
v4=np.array([50.9033,95.15,111.555,116.541,124.945,123.944,123.923,124.579,117.292,126.726,123.033,127.074,121.526,121.962,125.027])

# test_stencil_3d_range_vector
v5=np.array([59.5941,173.607,310.816,581.925,942.649,445.688,515.816,608.769,626.3,651.978,651.211,659.777,656.501,667.692,654.28])

# test_stencil_3d_range_hierarchical
v6=np.array([85.2335,272.356,527.502,671.682,1165.05,673.473,722.692,616.079,619.154,669.159,644.862,674.471,645.714,665.677,645.556])

# test_stencil_3d_range_hierarchical_linearized
v7=np.array([99.7289,285.45,554.959,693.982,1215.35,641.189,679.77,594.682,626.161,664.91,658.98,687.508,649.582,669.921,650.56])

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
