import numpy as np

# ソフトマックス関数
# 分類問題の出力層に使用される活性化関数である。
# n次元実数ベクトルを確率に落とし込んでいる。
def softmax(input):
	# オーバーフロー対策を行う。
	# 最も大きい値を取得する。
	maxNum = np.max(input)
	# 各要素から最も大きい値を引く。
	inputAfterOverflowMeasure = np.exp(input - maxNum)
	# ソフトマックス関数の分母
	sumInput = np.sum(inputAfterOverflowMeasure)

	return inputAfterOverflowMeasure / sumInput
