import numpy as np
import matplotlib.pyplot as plt

hls = np.loadtxt("hls/output_dcganW4A4.txt")
pt = np.loadtxt("pt_output.txt")

# hls = np.loadtxt("hls/out.txt")
# pt = np.loadtxt("pt_output.txt")

if (len(pt) != len(hls)):
	raise Exception("Shape not same")
diff = pt-hls
if not np.allclose(pt, hls):
	print("ERROR")
	print(max(diff), min(diff), diff)
	print(np.count_nonzero(diff))

plt.plot(pt, hls, 'o')
plt.show()

