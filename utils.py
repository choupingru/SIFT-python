import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
from skimage import filters
from matplotlib.patches import Circle

def make_pyramid(image, pyramid_depth, octave_depth, init_sigma, k):
	pyramids = [[]]
	sigma = init_sigma
	current_h, current_w = image.shape[:2]

	# first pyramid
	for first in range(octave_depth):
		gaued = image.copy()
		gaued = ndimage.filters.gaussian_filter(gaued, sigma)
		pyramids[-1].append(gaued)
		sigma *= k
	pyramids[-1] = np.array(pyramids[-1])

	# first S were downsample from previous layer, other were create by gaussian
	for p_depth in range(1, pyramid_depth):
		current_w, current_h = current_w // 2, current_h//2
		image = cv2.resize(image, (current_w, current_h))
		pyramids.append([])
		sigma = init_sigma * (2 ** p_depth)
		for o_depth in range(octave_depth):
			if o_depth < 3:
				pyramids[-1].append(cv2.resize(pyramids[-2][-3+o_depth], (current_w, current_h)))
			else:
				gaued = image.copy()
				gaued = ndimage.filters.gaussian_filter(gaued, sigma)
				pyramids[-1].append(gaued)
			sigma *= k
		pyramids[-1] = np.array(pyramids[-1])
	return pyramids

def DoG(pyramid, depth, octave_level):
	output = []
	for i in range(depth):
		output.append([])
		for j in range(octave_level-1):
			diff = pyramid[i][j+1] - pyramid[i][j]
			output[-1].append(diff)
		output[-1] = np.array(output[-1])
	return output

def find_extreme(dog_pyramid, depth, octave_level, contrast_threshold):
	keypoints = []
	
	for pyramid_index in range(depth):
		minmax_x, minmax_y = [], []	
		w, h = dog_pyramid[pyramid_index][0].shape[:2]
		cnt = 0
		for octave_index in range(1, octave_level-2):
			min_map = ndimage.minimum_filter(dog_pyramid[pyramid_index][octave_index-1:octave_index+2], size=(3, 3, 3))
			min_map = (dog_pyramid[pyramid_index][octave_index] == min_map[1])
			max_map = ndimage.maximum_filter(dog_pyramid[pyramid_index][octave_index-1:octave_index+2], size=(3, 3, 3))
			max_map = (dog_pyramid[pyramid_index][octave_index] == max_map[1])
			minmax_map = np.logical_or(min_map, max_map)
			indexes = np.where(minmax_map == 1)
			if not np.any(indexes[0]):
				continue
			minmax_x, minmax_y = indexes
			for (x, y) in zip(minmax_x, minmax_y):
				if x < w-1 and y < h-1 and x > 0 and y > 0:
					point = localizedQuadratic(dog_pyramid, pyramid_index, octave_index, x, y, w, h, octave_level, contrast_threshold)
					if point:
						keypoints.append(point)
	return keypoints

def localizedQuadratic(dog_pyramid, p_index, o_index, row_index, col_index, width, height, octave_level, contrast_threshold):

	counter = 0
	while counter < 5:
		cube = dog_pyramid[p_index][o_index-1:o_index+2, row_index-1:row_index+2, col_index-1:col_index+2]
		gradient_map = gradientMatrix(cube)
		hessian_map = hessianMatrix(cube)
		offset = np.linalg.lstsq(hessian_map, -gradient_map, rcond=-1)[0]
		if np.all(abs(offset) < 0.5):
			break
		else:
			row_index += int(offset[0])
			col_index += int(offset[1])
			o_index += int(offset[2])
		counter += 1
		if row_index <= 0 or col_index <= 0 or o_index <= 0 or row_index >= width-1 or col_index >= height-1 or o_index >= octave_level-2:
			return None # outside the image
	new_value = cube[1, 1, 1] + 0.5 * gradient_map @ offset
	if abs(new_value)  < contrast_threshold:
		return None # low contrast point
	else:
		hessian_trace = np.trace(hessian_map[:2, :2])
		hessian_determine = np.linalg.det(hessian_map[:2, :2])
		if hessian_determine == 0: 
			return None
		if (hessian_trace ** 2) / hessian_determine > 12.1:
			return None # edge
	return (p_index, o_index, 1.6 * (2 ** p_index) * (2 ** (o_index/3)), row_index, col_index)
	
def keypointDescriptor(dog_pyramid, keypoint, window_size=8):
	p_index, o_index, sigma, x, y = point
	x, y = x * (2 ** (-1 + p_index)), y * (2 ** (-1 + p_index))
	sigma = 1.6 * (2 ** p) * (2 **(o_index / 3))
	weight = 1 / (sigma**2)
	image = dog_pyramid[p_index][o_index]
	features_36 = []
	for i in range(-window_size, window_size):
		for j in range(-window_size, window_size):
			magnitude = ((image[i+x+1, j+y] - image[i+x-1, j+y]) ** 2 + (image[i+x, j+y+1] - image[i+x, j+y-1]) ** 2) ** 0.5
			angle = (np.degrees(np.arctan((image[i+x, j+y+1] - image[i+x, j+y-1]) / (image[i+x+1, j+y] - image[i+x-1, j+y]))) + 360) % 360
			weight = weight * np.exp(-(i**2 + j**2) / (sigma ** 2)) / (1.5 * window_size * 2)
			angle_bins = [0] * 36
			angle_bins[angle // 10] += weight * magnitude
			features_36.append(angle_bins)
	features_36 = np.array(features_36)
	features_8 = []
	for i in range(0, window_size*2, window_size//2):
		for j in range(0, window_size*2, window_size//2):
			region = features_36[i:i+window_size//2, j:j+window_size//2]
			########### 8 bins feature
	return angle_bins


def gradientMatrix(cube):
	dx = (cube[1, 1, 2] - cube[1, 1, 0]) / 2
	dy = (cube[1, 2, 1] - cube[1, 0, 1]) / 2
	ds = (cube[2, 1, 1] - cube[0, 1, 1]) / 2
	return np.array([dx, dy, ds])

def hessianMatrix(cube):

	center_pixel = cube[1, 1, 1]
	dxx = (cube[1, 1, 2] - 2 * center_pixel + cube[1, 1, 0])
	dyy = (cube[1, 2, 1] - 2 * center_pixel + cube[1, 0, 1])
	dss = (cube[2, 1, 1] - 2 * center_pixel + cube[0, 1, 1])
	dxy = (cube[1, 2, 2] - cube[1, 0, 2] - cube[1, 2, 0] + cube[1, 0, 0]) / 4
	dxs = (cube[2, 1, 2] - cube[2, 1, 0] - cube[0, 1, 2] + cube[0, 1, 0]) / 4
	dys = (cube[2, 2, 1] - cube[2, 0, 1] - cube[0, 2, 1] + cube[0, 0, 1]) / 4
	return np.array([[dxx, dxy, dxs],
					 [dxy, dyy, dys],
					 [dxs, dys, dss]])


if __name__ == '__main__':

	image = cv2.imread('./lena.png')
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = image.astype(np.float32)
	image = (image - image.min()) / (image.max() - image.min())
	image_up = cv2.resize(image, (0, 0), fx=2, fy=2)

	pyramid = make_pyramid(image_up, 3, 6, 1.6, 2**(1/3))
	dog = DoG(pyramid, depth=3, octave_level=6)
	keys = find_extreme(dog, 3, 6, 0.05)
	fig,ax = plt.subplots(1)
	ax.imshow(image, cmap='gray')

	for k in keys:
		p, _, _, r, c = k
		
		c, r = c * (2**(-1+p)), r * (2**(-1+p))
		circ = Circle((c,r),3)
		ax.add_patch(circ)


	plt.show()


