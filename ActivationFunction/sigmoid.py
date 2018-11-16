import numpy as np

# シグモイド関数
# y = 1 / (1 + exp(-x))
# 不連続であるステップ関数を滑らかな関数に近似する。
def sigmoid(input):
	return 1 / (1 + np.exp(-input))