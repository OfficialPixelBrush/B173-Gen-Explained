import numpy as np
from java_random import JavaRandom
from java_math import JavaMath

class PerlinGenerator:
	K_GRAD3_U = [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 ]
	K_GRAD3_V = [ 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2 ]
	K_GRAD2_U = [ 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2 ]
	K_GRAD2_V = [ 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1 ]
	K_SIGN_BIT0 = [ 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1 ]
	K_SIGN_BIT1 = [ 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1 ]


	def __init__(self, rand : JavaRandom):
		self.permutations : np.int32 = [0] * 512
		self.coordX = rand.next_double() * 256.0
		self.coordY = rand.next_double() * 256.0
		self.coordZ = rand.next_double() * 256.0
		for i in range(256):
			self.permutations[i] = i
		
		for i in range(256):
			j = rand.next_int_bound(256 - i) + i
			self.permutations[i], self.permutations[j] = self.permutations[j], self.permutations[i]
			self.permutations[i + 256] = self.permutations[i]
	
	def grad_3d(self, hash : np.int32, x : np.float64, y : np.float64, z : np.float64 ):
		h : np.uint32 = np.uint32(hash) & 15
		comp = [x,y,z]
		u : np.float64 = comp[self.K_GRAD3_U[h]]
		v : np.float64 = comp[self.K_GRAD3_V[h]]
		return u * self.K_SIGN_BIT0[h] + v * self.K_SIGN_BIT1[h]
	
	def grad_2d(self, hash : np.int32, x : np.float64, y : np.float64 ):
		h : np.uint32 = np.uint32(hash) & 15
		comp = [x,y,0.0]
		u : np.float64 = comp[self.K_GRAD3_U[h]]
		v : np.float64 = comp[self.K_GRAD3_V[h]]
		return u * self.K_SIGN_BIT0[h] + v * self.K_SIGN_BIT1[h]
	
	def fade(self, value : np.float64 ):
		return value * value * value * (value * (value * np.float64(6.0) - np.float64(15.0)) + np.float64(10.0))
	
	def generate_noise_base(self, x: np.float64, y: np.float64, z: np.float64):
		x = x + self.coordX
		y = y + self.coordY
		z = z + self.coordZ

		xInt : np.int32 = JavaMath.f64_to_int32(x)
		yInt : np.int32 = JavaMath.f64_to_int32(y)
		zInt : np.int32 = JavaMath.f64_to_int32(z)
		if (x < xInt):
			xInt = xInt - 1
		if (y < yInt):
			yInt = yInt - 1
		if (z < zInt):
			zInt = zInt - 1
		
		xIndex : np.int32 = xInt & 255
		yIndex : np.int32 = yInt & 255
		zIndex : np.int32 = zInt & 255

		x = x - xInt
		y = y - yInt
		z = z - zInt

		w : np.float64 = self.fade(x)
		v : np.float64 = self.fade(y)
		u : np.float64 = self.fade(z)
		permXY : np.int32 = self.permutations[xIndex] + yIndex
		permXYZ : np.int32 = self.permutations[permXY] + zIndex
		permXY = self.permutations[permXY + 1] + zIndex
		xIndex = self.permutations[xIndex + 1] + yIndex
		yIndex = self.permutations[xIndex] + zIndex
		xIndex = self.permutations[xIndex + 1] + zIndex
		return JavaMath.lerp(u,
						JavaMath.lerp(v,
								JavaMath.lerp(w, self.grad_3d(self.permutations[permXYZ], x, y, z),
									self.grad_3d(self.permutations[yIndex], x - 1.0, y, z)),
								JavaMath.lerp(w, self.grad_3d(self.permutations[permXY], x, y - 1.0, z),
									self.grad_3d(self.permutations[xIndex], x - 1.0, y - 1.0, z))),
						JavaMath.lerp(v,
								JavaMath.lerp(w, self.grad_3d(self.permutations[permXYZ + 1], x, y, z - 1.0),
									self.grad_3d(self.permutations[yIndex + 1], x - 1.0, y, z - 1.0)),
								JavaMath.lerp(w, self.grad_3d(self.permutations[permXY + 1], x, y - 1.0, z - 1.0),
									self.grad_3d(self.permutations[xIndex + 1], x - 1.0, y - 1.0, z - 1.0))))

	def generate_noise_2d_raw(self, x : np.float64, y : np.float64):
		return self.generate_noise_base(x, y, 0.0)
	
	def generate_noise_3d_raw(self, x : np.float64, y : np.float64, z : np.float64):
		return self.generate_noise_base(x, y, z)
	
	def generate_noise_2d(self, data, x: np.float64, z: np.float64, size_x: np.int32, size_z: np.int32, scale_x: np.float64, scale_z: np.float64, amplitude: np.float64):
		invAmp : np.float64 = 1.0 / amplitude
		index = 0
		for ix in range(size_x):
			fx : np.float64 = (x + ix) * scale_x + self.coordX
			ix = np.int32(fx)
			if (fx < ix):
				ix = ix - 1
			px : np.int32 = ix & 255
			fx = fx - ix
			u : np.float64 = self.fade(fx)
			for iz in range(size_z):
				fz : np.float64 = (z + iz) * scale_z + self.coordZ
				iz = np.int32(fz)
				if (fz < iz):
					iz = iz - 1
				pz : np.int32 = iz & 255
				fz = fz - iz
				w : np.float64 = self.fade(fz)

				a  : np.int32 = self.permutations[px] + 0
				aa : np.int32  = self.permutations[a] + pz
				b  : np.int32 = self.permutations[px + 1] + 0
				ba : np.int32  = self.permutations[b] + pz

				i1 : np.float64 = JavaMath.lerp(u, self.grad_2d(self.permutations[aa], fx, fz), self.grad_3d(self.permutations[ba], fx - 1.0, 0.0, fz))

				i2 : np.float64 = JavaMath.lerp(u, self.grad_3d(self.permutations[aa + 1], fx, 0.0, fz - 1.0),
									self.grad_3d(self.permutations[ba + 1], fx - 1.0, 0.0, fz - 1.0))
				data[index] = data[index] + np.float64(JavaMath.lerp(w, i1, i2) * invAmp)
				index = index + 1
	
	def generate_noise_3d(self, data, x: np.float64, y: np.float64, z: np.float64, size_x: np.int32, size_y: np.int32, size_z: np.int32, scale_x: np.float64, scale_y: np.float64, scale_z: np.float64, amplitude: np.float64):
		invAmp : np.float64 = 1.0 / amplitude
		lastPermY : np.int32 = -1

		lerpAX : np.float64 = 0.0
		lerpBX : np.float64 = 0.0
		lerpAY : np.float64 = 0.0
		lerpBY : np.float64 = 0.0

		index = 0

		for xi in range(size_x):
			fx : np.float64 = (x + xi) * scale_x + self.coordX
			ix = np.int32(fx)
			if (fx < ix):
				ix = ix - 1
			px : np.int32 = ix & 255
			fx = fx - ix
			u : np.float64 = self.fade(fx)
			for zi in range(size_z):
				fz : np.float64 = (z + zi) * scale_z + self.coordZ
				iz = np.int32(fz)
				if (fz < iz):
					iz = iz - 1
				pz : np.int32 = iz & 255
				fz = fz - iz
				w : np.float64 = self.fade(fz)
				for yi in range(size_y):
					fy : np.float64 = (y + yi) * scale_y + self.coordY
					iy = np.int32(fy)
					if (fy < iy):
						iy = iy - 1
					py : np.int32 = iy & 255
					fy = fy - iy
					v : np.float64 = self.fade(fy)
					if (yi == 0 or py != lastPermY):
						lastPermY = py

						a  : np.int32 = self.permutations[px] + py
						aa : np.int32  =self.permutations[a] + pz
						ab : np.int32  =self.permutations[a + 1] + pz
						b  : np.int32 = self.permutations[px + 1] + py
						ba : np.int32  =self.permutations[b] + pz
						bb : np.int32  =self.permutations[b + 1] + pz

						lerpAX = JavaMath.lerp(u, self.grad_3d(self.permutations[aa], fx, fy, fz),
									  self.grad_3d(self.permutations[ba], fx - 1.0, fy, fz))
						lerpBX = JavaMath.lerp(u, self.grad_3d(self.permutations[ab], fx, fy - 1.0, fz),
									  self.grad_3d(self.permutations[bb], fx - 1.0, fy - 1.0, fz))
						lerpAY = JavaMath.lerp(u, self.grad_3d(self.permutations[aa + 1], fx, fy, fz - 1.0),
									  self.grad_3d(self.permutations[ba + 1], fx - 1.0, fy, fz - 1.0))
						lerpBY = JavaMath.lerp(u, self.grad_3d(self.permutations[ab + 1], fx, fy - 1.0, fz - 1.0),
									  self.grad_3d(self.permutations[bb + 1], fx - 1.0, fy - 1.0, fz - 1.0))
					i1 : np.float64 = JavaMath.lerp(v, lerpAX, lerpBX)
					i2 : np.float64 = JavaMath.lerp(v, lerpAY, lerpBY)
					data[index] = data[index] + np.float64(JavaMath.lerp(w, i1, i2) * invAmp)
					index = index + 1

class PerlinOctaves:
	generators = []

	def __init__(self, rand : JavaRandom, octaves : int):
		self.generators = []
		for i in range(octaves):
			self.generators.append(PerlinGenerator(rand))
	
	def generate_octaves_2d_raw(self, x: np.float64, y: np.float64) -> np.float64:
		value : np.float64 = 0.0
		scale : np.float64 = 1.0
		for g in self.generators:
			value = value + g.generate_noise_2d(x * scale, y * scale) / scale
			scale = scale / 2.0
		return value

	def generate_octaves_3d(self, data, x: np.float64, y: np.float64, z: np.float64, size_x: np.int32, size_y: np.int32, size_z: np.int32, scale_x: np.float64, scale_y: np.float64, scale_z: np.float64):
		max_size = size_x * size_y * size_z
		for i in range(max_size):
			data[i] = 0.0
		
		multi : np.float64 = 1.0
		for g in self.generators:
			g.generate_noise_3d(data, x,y,z, size_x, size_y, size_z, scale_x * multi, scale_y * multi, scale_z * multi, multi)
			multi = multi / 2.0

	def generate_octaves_2d(self, data, x: np.float64, y: np.float64, size_x: np.int32, size_y: np.int32, scale_x: np.float64, scale_y: np.float64):
		self.generate_octaves_3d(data, x, 10.0, y, size_x, 1, size_y, scale_x, 1.0, scale_y)