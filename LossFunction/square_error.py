import numpy as np

# 二乗和誤差
# teacher:教師信号、output:出力値
# 教師あり学習時に使用する損失関数
def square_error(teacher, output):
	return np.sum((np.square(teacher - output)) / 2.0)
