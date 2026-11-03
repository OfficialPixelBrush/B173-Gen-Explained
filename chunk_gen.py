import numpy as np
from java_random import JavaRandom
from java_math import JavaMath
from perlin_gen import PerlinGenerator, PerlinOctaves

CHUNK_WIDTH : np.int32 = 16
CHUNK_HEIGHT : np.int32 = 128
CHUNK_SUB_SAMPLE_SCALE : np.int32 = 4
CHUNK_SUB_SAMPLE_WIDTH : np.int32 = (CHUNK_WIDTH // CHUNK_SUB_SAMPLE_SCALE) + 1
CHUNK_SUB_SAMPLE_HEIGHT : np.int32 = (CHUNK_HEIGHT // CHUNK_SUB_SAMPLE_SCALE) + 1
CHUNK_SUB_SAMPLE_AREA : np.int32 = (CHUNK_SUB_SAMPLE_WIDTH * CHUNK_SUB_SAMPLE_WIDTH)
CHUNK_SUB_SAMPLE_VOLUME : np.int32 = (CHUNK_SUB_SAMPLE_WIDTH * CHUNK_SUB_SAMPLE_HEIGHT * CHUNK_SUB_SAMPLE_WIDTH)

class OverworldGenerator:

    def __init__(self, seed : np.int64):
        self.rng = JavaRandom(seed)
        self.lowGen = PerlinOctaves(self.rng, 16)
        self.highGen = PerlinOctaves(self.rng, 16)
        self.selectorGen = PerlinOctaves(self.rng, 8)
        self.sandGravelGen = PerlinOctaves(self.rng, 4)
        self.stoneGen = PerlinOctaves(self.rng, 4)
        self.contintenalnessGen = PerlinOctaves(self.rng, 10)
        self.depthGen = PerlinOctaves(self.rng, 16)
        # treeDensityGen = PerlinOctaves(self.rng, 8)
        # TODO: Biome gen
        # biomeGen = BiomeGenerator(seed);

        self.contintentalnessData = [0.0] * (CHUNK_SUB_SAMPLE_AREA)
        self.depthData = [0.0] * (CHUNK_SUB_SAMPLE_AREA)
        self.selectorData = [0.0] * (CHUNK_SUB_SAMPLE_VOLUME)
        self.lowData = [0.0] * (CHUNK_SUB_SAMPLE_VOLUME)
        self.highData = [0.0] * (CHUNK_SUB_SAMPLE_VOLUME)

    def generate_terrain_noise(self, x : np.int32, y : np.int32, z : np.int32, max_x : np.int32, max_y : np.int32, max_z : np.int32):
        densityData = [0.0] * (max_x * max_y * max_z)
        scale : np.float64 = 684.412
        self.contintenalnessGen.generate_octaves_2d(self.contintentalnessData, x, z, max_x, max_z, 1.121, 1.121)
        self.depthGen.generate_octaves_2d(self.depthData, x, z, max_x, max_z, 200.0, 200.0)
        
        f64_x : np.float64 = np.float64(x)
        f64_y : np.float64 = np.float64(y)
        f64_z : np.float64 = np.float64(z)
        self.selectorGen.generate_octaves_3d(self.selectorData, f64_x, f64_y, f64_z, max_x, max_y, max_z, scale / 80.0, scale / 160.0, scale / 80.0)
        self.lowGen.generate_octaves_3d(self.lowData, f64_x, f64_y, f64_z, max_x, max_y, max_z, scale, scale, scale)
        self.highGen.generate_octaves_3d(self.highData, f64_x, f64_y, f64_z, max_x, max_y, max_z, scale, scale, scale)

        xyzIndex : np.int32 = 0
        xzIndex : np.int32 = 0

        scaleFraction : np.int32 = CHUNK_WIDTH // max_x
        for ix in range(max_x):
            sampleX : np.int32 = ix * scaleFraction + scaleFraction / 2
            for iz in range(max_z):
                sampleZ : np.int32 = iz * scaleFraction + scaleFraction / 2
                # Sample the 2D noises first
                sampleIndex = sampleX * CHUNK_WIDTH + sampleZ
                temp : np.float64 = 0.5 # TODO: Temperature Array
                humi : np.float64 = 0.5 * temp # TODO: Humidity Array
                humi = 1.0 - humi
                humi = humi * humi
                humi = humi * humi
                humi = 1.0 - humi
                # Get continentalness
                cont : np.float64 = (self.contintentalnessData[xzIndex] + 256.0) / 512.0
                cont = cont * humi
                cont = min(cont, 1.0)
                # Get depth
                depth : np.float64 = self.depthData[xzIndex] / 8000.0
                if (depth < 0.0):
                    depth = -depth * 0.3
                depth = depth * 3.0 - 2.0
                if (depth < 0.0):
                    depth = depth / 2.0
                    depth = max(depth, -1.0)
                    depth = depth / 1.4
                    depth = depth / 2.0
                    cont = 0.0
                else:
                    depth = min(depth, 1.0)
                    depth = depth / 8.0
                # Avoid monoliths
                if (cont < 0.0):
                    cont = 0.0
                cont = cont + 0.5
                depth = depth * np.float64(max_y) / 16.0

                elevOffset = np.float64(max_y) / 2.0 + depth * 4.0
                xzIndex = xzIndex + 1

                for iy in range(max_y):
                    # Sample the 3D noises
                    density : np.float64 = 0.0
                    densityOffset : np.float64 = (np.float64(iy) - elevOffset) * (12.0 / cont)
                    if (densityOffset < 0.0):
                        densityOffset = densityOffset * 4.0
                    
                    # Figure out whether low or high noise should proliferate
                    low = self.lowData[xyzIndex] / 512.0
                    high = self.highData[xyzIndex] / 512.0
                    select = (self.selectorData[xyzIndex] / 10.0 + 1.0) / 2.0
                    if (select < 0.0):
                        density = low
                    elif (select > 1.0):
                        density = high
                    else:
                        density = JavaMath.lerp(select, low, high)
                    
                    density -= densityOffset
                    if (iy > max_y - 4):
                        heightFade : np.float64 = np.float64(np.float32(iy - (max_y - 4)) / np.float32(3.0))
                        density = (density * (1.0 - heightFade)) + (-10.0 * heightFade)
                    # Pass the density on
                    densityData[xyzIndex] = density
                    xyzIndex = xyzIndex + 1
        return densityData
                    


    def generate_terrain(self, x : np.int32, z : np.int32):
        self.generate_terrain_noise(x * CHUNK_SUB_SAMPLE_SCALE, 0, z * CHUNK_SUB_SAMPLE_SCALE, CHUNK_SUB_SAMPLE_WIDTH, CHUNK_SUB_SAMPLE_HEIGHT, CHUNK_SUB_SAMPLE_WIDTH)
        return

    def generate_chunk(self, x : np.int32, z : np.int32):
        self.rng.set_seed((np.int64(x) * 341873128712) + (np.int64(z) * 132897987541))

        # Biome generation

        #print(self.generate_terrain_noise(x * CHUNK_SUB_SAMPLE_SCALE, 0, z * CHUNK_SUB_SAMPLE_SCALE, CHUNK_SUB_SAMPLE_WIDTH, CHUNK_SUB_SAMPLE_HEIGHT, CHUNK_SUB_SAMPLE_WIDTH))