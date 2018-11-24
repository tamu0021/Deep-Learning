import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient
sys.path.append('./ActivationFunction')
import sigmoid as sig
import softmax as sm
sys.path.append('./LossFunction')
import cross_entropy_error as cee

# 2層ニューラルネットワークのクラス
class TwoLayerNet:

	# コンストラクタ
	# 重みとバイアスの値を初期化する。
	def __init__(	self, inputSize, hiddenSize, outputSize, 
					weightInitStd = 0.01):
		self.params = {}
		self.params['W1'] = weightInitStd * \
							np.random.randn(inputSize, hiddenSize)
		self.params['b1'] = np.zeros(hiddenSize)
		self.params['W2'] = weightInitStd * \
							np.random.randn(hiddenSize, outputSize)
		self.params['b2'] = np.zeros(outputSize)

	# 活性化関数を用いて入力層から出力層までの演算を行う。
	def predict(self, inputParam):
		# 重みの代入
		weight1 = self.params['W1']
		weight2 = self.params['W2']
		# バイアスの代入
		bias1 = self.params['b1']
		bias2 = self.params['b2']

		# 入力層から隠れ層まで
		# 活性化関数としてシグモイド関数を用いる。
		inputToHidden = np.dot(inputParam, weight1) + bias1
		hiddenParam = sig.sigmoid(inputToHidden)
		# 隠れ層から出力層まで
		# 活性化関数としてソフトマックス関数を用いる。
		hiddenToOutput = np.dot(hiddenParam, weight2) + bias2
		outputParam = sm.softmax(hiddenToOutput)

		return outputParam

	# 損失の演算
	# 交差エントロピー誤差を用いる。
	def loss(self, inputParam, teacher):
		outputParam = self.predict(inputParam)

		return cee.cross_entropy_error(outputParam, teacher)

	# 正確な値となっている配列数の割合を返す。
	def accuracy(self, inputParam, teacher):
		outputParam = self.predict(inputParam)
		outputParam = np.argmax(outputParam, axis = 1)
		teacher = np.argmax(teacher, axis = 1)

		# output = teacherとなっている配列数の割合を返す。
		accuracy = np.sum(outputParam == teacher) / float(inputParam.shape[0])

		return accuracy

	# 勾配の計算を行う。
	def numerical_gradient(self, inputParam, teacher):
		lossWeight = lambda weight: self.loss(inputParam, teacher)

		grads = {}
		grads['W1'] = numerical_gradient(lossWeight, self.params['W1'])
		grads['W2'] = numerical_gradient(lossWeight, self.params['b1'])
		grads['b1'] = numerical_gradient(lossWeight, self.params['W2'])
		grads['b2'] = numerical_gradient(lossWeight, self.params['b2'])

		return grads

