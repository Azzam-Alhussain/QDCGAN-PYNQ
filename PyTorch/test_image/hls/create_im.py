import numpy as np
import matplotlib.pyplot as plt
import itertools
examples = 1
test_images = np.loadtxt('output_celebA-W4A4.txt')
test_images = test_images.reshape(examples*examples, 64,64,3)


if examples > 1:
	size_figure_grid = examples
	fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(examples,examples))
	for i,j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
	    ax[i,j].get_xaxis().set_visible(False)
	    ax[i,j].get_yaxis().set_visible(False)

	for k in range(examples*examples):
	    i = k//examples
	    j = k%examples
	    single_image = test_images[k]
	    single_image = (((single_image - single_image.min()) * 255) / (single_image.max() - single_image.min())).astype(np.uint8)
	    ax[i,j].cla()
	    ax[i,j].imshow(single_image, cmap=None)
	plt.savefig("hls.png")
	plt.close()

else:
	single_image = test_images[0]
	single_image = (((single_image - single_image.min()) * 255) / (single_image.max() - single_image.min())).astype(np.uint8)
	from PIL import Image
	from matplotlib import cm
	im = Image.fromarray(np.uint8(single_image))
	im.save('hls.png')
