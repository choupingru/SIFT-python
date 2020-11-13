import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
from skimage import filters

def make_octave(input, level, sigma, k):
	if level == 0:
		return []
	else:
		w, h = input.shape[:2]
		gaued = input.copy()
		gaued = ndimage.filters.gaussian_filter(input, sigma)
		return [gaued] + make_octave(input, level-1, sigma * k, k)

def make_pyramid(input, depth, octave_level, sigma):
	if depth == 0:
		return []
	else:
		current_layer = make_octave(input, octave_level, sigma, 2 ** 0.5)
		h, w = input.shape[:2]
		downsample = cv2.resize(input, (w // 2, h // 2))
		return [current_layer] + make_pyramid(downsample, depth - 1, octave_level, sigma * 2)

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
		
		w, h = dog_pyramid[pyramid_index][0].shape[:2]
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
			for index, (x, y) in enumerate(zip(minmax_x, minmax_y)):
				# keypoints.append((pyramid_index, octave_index, x, y))
				if x < w-1 and y < h-1 and x > 0 and y > 0:
					point = localizedWithQuadraticFunction(pyramid_index, octave_index, x, y, w, h, octave_level, contrast_threshold)
					if point:
						keypoints.append(point)
	return keypoints

def localizedWithQuadraticFunction(p_index, o_index, row_index, col_index, width, height, octave_level, contrast_threshold):

	low_contrast, edge, out_of_index = False, False, False
	counter = 0
	while counter < 3:
		cube = dog_pyramid[p_index][o_index-1:o_index+2, row_index-1:row_index+2, col_index-1:col_index+2]
		gradient_map = gradientMatrix(cube)
		hessian_map = hessianMatrix(cube)
		offset = -np.linalg.lstsq(hessian_map, gradient_map, rcond=-1)[0]
		if np.all(abs(offset) < 0.5):
			break
		else:
			row_index += int(offset[0])
			col_index += int(offset[1])
			o_index += int(offset[2])
		counter += 1
		if row_index <= 0 or col_index <= 0 or o_index <= 0 or row_index >= width-1 or col_index >= height-1 or o_index >= octave_level-2:
			out_of_index = True
			break

	new_value = cube[1, 1, 1] + 0.5 * np.dot(gradient_map, offset)
	if abs(new_value) * 3 < contrast_threshold:
		low_contrast = True
	else:
		hessian_trace = np.trace(hessian_map[:2, :2])
		hessian_determine = np.linalg.det(hessian_map[:2, :2])
		if hessian_determine == 0: 
			return None
		# print((hessian_trace ** 2) / hessian_determine)
		if (hessian_trace ** 2) / hessian_determine < 12.1:
			edge = False
		else:
			edge = True
	if not low_contrast and not out_of_index and not edge:
		return (p_index, o_index, row_index, col_index)
	else:
		return None

def gradientMatrix(cube):
	dx = (cube[1, 1, 2] - cube[1, 1, 0]) / 2
	dy = (cube[1, 0, 1] - cube[1, 2, 1]) / 2
	ds = (cube[2, 1, 1] - cube[0, 1, 1]) / 2
	return np.array([dx, dy, ds])

def hessianMatrix(cube):

	center_pixel = cube[1, 1, 1]
	dxx = (cube[1, 1, 2] - 2 * center_pixel + cube[1, 1, 0])
	dyy = (cube[1, 0, 1] - 2 * center_pixel + cube[1, 2, 1])
	dss = (cube[2, 1, 1] - 2 * center_pixel + cube[0, 1, 1])
	dxy = (cube[1, 0, 2] - cube[1, 2, 2] - cube[1, 0, 0] + cube[1, 2, 0]) / 4
	dxs = (cube[2, 1, 2] - cube[2, 1, 0] - cube[0, 1, 2] + cube[0, 1, 1]) / 4
	dys = (cube[2, 0, 1] - cube[2, 2, 1] - cube[0, 0, 1] + cube[0, 2, 1]) / 4
	return np.array([[dxx, dxy, dxs],
					 [dxy, dyy, dys],
					 [dxs, dys, dss]])


if __name__ == '__main__':

	image = cv2.imread('./test1.jpg')
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = image.astype(np.float32) / 255


	octave_level = 5
	depth = 2
	pyramid = make_pyramid(image, depth=depth, octave_level=octave_level, sigma=1.6)
	dog_pyramid = DoG(pyramid, depth=depth, octave_level=octave_level)

	keys = find_extreme(dog_pyramid, depth, octave_level, 0.03)
	
	plt.imshow(image, cmap='gray')
	plt.show()


