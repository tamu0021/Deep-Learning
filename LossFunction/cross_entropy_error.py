import numpy as np

# 交差エントロピー誤差
# teacher:教師信号、output:出力値
# 教師あり学習時に使用する損失関数
def cross_entropy_error(teacher, output):
	# log(0)にならないよう、十分小さい数δを導入する。
	delta = 1e-7
	return -np.sum(teacher * np.log(output + delta))