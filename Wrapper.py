#!/usr/bin/evn python

# AUTHOR: Pradnya Sushil Shinde 

import os 
import cv2 
import numpy as np 
import argparse
import scipy.optimize as opt
import copy

# x = A [R t]X
# x -- 2D coords
# X -- 3D coords
# R -- Rotational Mat
# T -- Translational Mat
# A -- Camera Intrinsic Mat 

class AutoCalibrator():
	def __init__(self, cal_imgs_path, r, c, sq_size):
		"""
		Initialize Auto Calibrator.

		Parameters:
		- cal_imgs_path (str): Path to calibration images.
		"""
		self.cal_imgs_path = cal_imgs_path
		self.row = r 
		self.col = c 
		self.sq_size = sq_size

	def readImgs(self):
		cal_imgs_list = os.listdir(self.cal_imgs_path)
		cal_imgs_list.sort()

		print("Calibration Images List: ", cal_imgs_list)

		cal_imgs = []
		for img in cal_imgs_list:
			cal_imgs.append(cv2.imread(self.cal_imgs_path + img))

		# print("IMAGES: ", len(cal_imgs))
		return cal_imgs
	
	def computeCorners(self, imgs):
		img_coords = []
		# print("IMAGES: ", imgs.shape())
		save_corner_imgs = './ImgCorners/'
		if not os.path.isdir(save_corner_imgs):
			os.makedirs(save_corner_imgs)

		for i, img in enumerate(imgs):
			img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			ret, img_corners = cv2.findChessboardCorners(img_gray, (self.row, self.col), None)
			if ret == True:
				img_corners = np.reshape(img_corners, (-1,2))
				img_coords.append(img_corners)
				img_w_corners = cv2.drawChessboardCorners(img, (self.row, self.col), img_corners, True)
				cv2.imwrite(save_corner_imgs+str(i+1)+'.jpg', img_w_corners)
	
		print('Images with detected chessborard corners saved!')
		img_coords = np.array(img_coords)
		# print('CORNERS SHAPE', )

		return img_coords
	
	def computeWorldCoords(self):

		X, Y = np.meshgrid(range(self.row), range(self.col))

		X = X.reshape((self.row*self.col), 1)
		Y = Y.reshape((self.row*self.col), 1)

		world_coords = np.array(np.hstack((X, Y))*self.sq_size)
		# print("World Coordinates: ",world_coords)
		

		return world_coords

	# def getHomography(self, img_coords, world_coords):
	# 	H_mat = []

	# 	for i, coords in enumerate(img_coords):
	# 		H, mask = cv2.findHomography(world_coords, coords, cv2.RANSAC, 5.0)
	# 		H_mat.append(H)
	# 		# print('Homography Matrix for Image:' + str(i+1) + ": ", H)

	# 	return H_mat
	
	def getHomography(self, img_coords, world_coords):
			H_mat = []
			for i, coords in enumerate(img_coords):
					H_int = []
					for j in range(len(coords)):
						X, Y = world_coords[j]
						x, y = img_coords[i][j]
						H_int.append([X, Y, 1, 0, 0, 0, -X*x, -Y*x, -x])
						H_int.append([0, 0, 0, X, Y, 1, -X*y, -Y*y, -y])
					U, S, V = np.linalg.svd(np.asarray(H_int))
					H = V[-1, :]/V[-1, -1]
					H = H.reshape(3,3)
					H_mat.append(H)
				# print('Homography Matrix for Image:' + str(i+1) + ": ", H)
			return H_mat
	
	def computeV(self, H, i, j):
	
		v = np.array([
			H[i][0]*H[j][0],
			H[i][0]*H[j][1] + H[i][1]*H[j][0],
			H[i][1]*H[j][1],
			H[i][2]*H[j][0] + H[i][0]*H[j][2],
			H[i][2]*H[j][1] + H[i][1]*H[j][2],
			H[i][2]*H[j][2]
		])

		return v

	def computeB(self, H_mat):
		V = []
		for i in range(len(H_mat)):
			H_i = H_mat[i]
			H_i = H_i.T 
			v12 = self.computeV(H_i, 0, 1)
			v11 = self.computeV(H_i, 0, 0)
			v22 = self.computeV(H_i, 1, 1)
			v11_v22 = v11.T - v22.T
			V.append(v12)
			V.append(v11_v22)

		V = np.array(V)
		_, _, V_mat = np.linalg.svd(V, full_matrices=True)
		V_mat = V_mat.T
		B = V_mat[:, -1]

		return B
	
	def computeA(self, B_mat):
		v_0 = (B_mat[1]*B_mat[3] - B_mat[0]*B_mat[4])/(B_mat[0]*B_mat[2] - B_mat[1]**2)
		Lambda = B_mat[5] - (B_mat[3]**2 + v_0*(B_mat[1]*B_mat[3]-B_mat[0]*B_mat[4]))/B_mat[0]
		alpha = np.sqrt(Lambda/B_mat[0])
		beta = np.sqrt(Lambda*B_mat[0]/(B_mat[0]*B_mat[2] - B_mat[1]**2))
		gamma = (-B_mat[1]*(alpha**2)*beta)/Lambda
		u_0 = ((gamma*v_0)/beta) - ((B_mat[3]*(alpha**2))/Lambda)

		A = np.array([[alpha, gamma, u_0],
				[0, beta, v_0],
				[0, 0, 1]
				 ])
		
		return A
	def computeRt(self, H_mat, A):
		Rt = []
		for H in H_mat:
			Lambda = 1/np.linalg.norm((np.dot(np.linalg.inv(A), H[:,0])), ord=2)

			r1 = Lambda*np.dot(np.linalg.inv(A), H[:, 0])
			r2 = Lambda*np.dot(np.linalg.inv(A), H[:, 1])
			r3 = np.cross(r1, r2)
			t = Lambda*np.dot(np.linalg.inv(A), H[:, 2])

			Rt_i = np.transpose(np.vstack((r1, r2, t)))
			Rt.append(Rt_i)

		return Rt

	def get_params(self, A_int, kdist_init):
		x0 = np.array([A_int[0][0], 
				 A_int[0][1],
				 A_int[1][1],
				 A_int[0][2],
				 A_int[1][2],
				 kdist_init[0],
				 kdist_init[0]
				 ])
		return x0
	
	def ErrFunc(self, params, Rt_ext, img_coords, world_coords, func_call = 'Exclude Projection Coords'): #Error Minimization Function

		alpha, gamma, beta, u_0, v_0, k1, k2 = params
		A_int = np.array([[alpha, gamma, u_0],
				   [0, beta, v_0], 
				   [0, 0, 1]])
		k_dist = np.array([k1, k2])
		errorTot = []
		coordsTot = []
		for i, img_coord in enumerate(img_coords):
			img_coord_i = img_coord
			H_i = np.dot(A_int, Rt_ext[i])
			err_i = 0
			reproj_coords = []

			for j in range(len(img_coord_i)):
				world_coord_j = world_coords[j]

				M_hat = np.array([[world_coord_j[0]],
					  [world_coord_j[1]],
					  [1]])

				
				m = np.array([[img_coord_i[j][0]], 
					  [img_coord_i[j][1]],
					  [1]
					  ])
				
				proj_coords = np.matmul(Rt_ext[i], M_hat)
				proj_coords = proj_coords/proj_coords[2]
				x, y = proj_coords[0], proj_coords[1]

				img_plane_coords = np.matmul(H_i, M_hat)
				img_plane_coords = img_plane_coords/img_plane_coords[2]
				u, v = img_plane_coords[0], img_plane_coords[1]

				r = (np.square(x) + np.square(y))
				u_hat = u + (u-u_0)*(k_dist[0]*r + k_dist[1]*(np.square(r))) 
				v_hat = v + (v-v_0)*(k_dist[0]*r + k_dist[1]*(np.square(r))) 

				m_hat = np.array([u_hat,v_hat,[1]])
				reproj_coords.append(m_hat)			
				err = np.linalg.norm((m - m_hat), ord=2)	
				err_i +=err
			
			coordsTot.append(reproj_coords)
			errorTot.append(err_i/len(world_coords))

		coordsTot = np.array(coordsTot)
		errorTot = np.array(errorTot)

		if func_call == 'Include Projection Coords':
			return errorTot, coordsTot
		elif func_call == 'Exclude Projection Coords':
			return errorTot

	def plot_proj_pt(self, img, pts, img_num):
		save_op_imgs = './OutputImgs/'
		if not os.path.isdir(save_op_imgs):
			os.makedirs(save_op_imgs)
		for i in range(len(pts)):
			img_w_proj_pts = cv2.circle(img, (int(pts[i][0]), int(pts[i][1])), 7, (0, 0, 255), -2)
			cv2.imwrite(save_op_imgs+str(img_num)+'.jpg', img_w_proj_pts)

		

	
def main():
	r = 9 # number of rows representing inside corners
	c = 6 # number of columns representing inside corners
	sq_size = 21.5 # size of each square on the checkerboard

	cal_imgs_path = './Calibration_Imgs/'

	# Initialise class object
	auto_calib = AutoCalibrator(cal_imgs_path, r, c, sq_size)

	# Read images
	cal_imgs = auto_calib.readImgs()

	cal_imgs_org = copy.deepcopy(cal_imgs)
	imgs = cal_imgs.copy()

	# Get chessboard corners 
	img_coords = auto_calib.computeCorners(imgs)


	# Compute world coordinates based on he h,w, square size paramerts related to Checkerboard
	world_coords = auto_calib.computeWorldCoords()
	# print("Num of 3D Coords: ", len(world_coords))

	# Find homography between image coorinates of individual images and the claculated world points
	# print("Estimating homography.")
	H_mat = auto_calib.getHomography(img_coords, world_coords)
	# print("Homography Matrix: ", H_mat)

	# print("Computing B matrix")
	B = auto_calib.computeB(H_mat)
	# print("B matrix: ", B)

	# print("Computing Camera Intrinsic Matrix.")
	A_int_init = auto_calib.computeA(B) # A matrix, called as the camera intrinsic matrix
	print("Camera Intrinsic Matrix K: ", A_int_init)

	# print("Computing Camera Extrinsic Matrix.")
	Rt_ext = auto_calib.computeRt(H_mat, A_int_init)
	print("Camera Extrinsic Matrix: ", Rt_ext)

	k_dist_init = np.array([0, 0])
	print("Initial Distortion Values: ", k_dist_init)

	x0 = auto_calib.get_params(A_int_init, k_dist_init)
	print("Init Paramters: ", x0)

	pre_opt_error, pre_opt_coords = auto_calib.ErrFunc(x0, Rt_ext, img_coords, world_coords, func_call = 'Include Projection Coords')
	print("Pre-Optimization Reprojection Error: ", pre_opt_error)

	print("Optimizing to minimize error!")
	res = opt.least_squares(fun=auto_calib.ErrFunc, x0=x0, method='lm', args=[Rt_ext, img_coords, world_coords])

	x = res.x
	print("Optimized Parameters: ", x)
	
	post_opt_error, post_opt_coords = auto_calib.ErrFunc(x, Rt_ext, img_coords, world_coords, func_call = 'Include Projection Coords')
	print("Post-Optimization Reprojection Error: ", post_opt_error)
	# print("Total Post-Optimization Coords: ", len(post_opt_coords[0]))

	mean_err = np.mean(post_opt_error)
	print("Mean reprojectrion error is: ", mean_err)

	alpha, gamma, beta, u0, v0, k1_opt, k2_opt = x

	k_dist_opt = np.array([k1_opt, k2_opt, 0, 0, 0], dtype=float)
	print("Optimized Distortion Values: ", k_dist_opt)
	
	A_int_opt = np.array([[alpha, gamma, u0],
				[0, beta, v0],
				[0, 0, 1]
				 ])
	print("Optimised Camera Intrinsic Matrix: ", A_int_opt)

	print("Plotting and saving output result images")
	for i in range(len(cal_imgs_org)):
		img = cal_imgs_org[i]

		img = cv2.undistort(img, A_int_opt, k_dist_opt)
		save_undist_imgs = './RectifiedImages/'
		if not os.path.isdir(save_undist_imgs):
			os.makedirs(save_undist_imgs)
		cv2.imwrite(save_undist_imgs + str(i+1) + '.jpg', img)

		auto_calib.plot_proj_pt(img, post_opt_coords[i], i+1)

	print("Successfully saved images to the results folder")




main()


		