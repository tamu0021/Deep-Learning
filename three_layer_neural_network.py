import sys
sys.path.append('./ActivationFunction')
import numpy as np
import sigmoid as sig
import softmax as sofm

# 三層ニューラルネットワークのクラス
class Three_Layer_Neural_Network:

	# 仮に値を設定しておく。所謂初期化。
	# 層に入力される値
	input	= np.array([1.0, 0.5])
	# 重み
	weight = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
	# バイアス
	bias	= np.array([0.1, 0.2, 0.3])

	# コンストラクタ
	def __init__(self, input1, weight1, bias1):
		self.input		= input1
		self.weight		= weight1
		self.bias		= bias1

	# 新たな層の重みとバイアスを設定する。
	def set_weight_and_bias(self, weightx, biasx):
		self.weight = weightx
		self.bias 	= biasx

	# シグモイド関数を用いて、層に入力された値と重み、バイアスから次の層へ入力する値を得る。
	def input_to_layer_by_sigmoid(self):
		signal1 = np.dot(self.input, self.weight) + self.bias
		self.input = sig.sigmoid(signal1)

	# 恒等関数を用いて、層に入力された値と重み、バイアスから次の層へ入力する値を得る。
	def input_to_layer_by_identify(self):
		self.input = np.dot(self.input, self.weight) + self.bias

	# 結果を出力する。
	def print_result(self):
		print(self.input)
		print(self.weight)
		print(self.bias)

# 以下は使用例。
input1	= np.array([1.0, 0.5])
weight1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
bias1	= np.array([0.1, 0.2, 0.3])

threeLayerNeuralNetwork = Three_Layer_Neural_Network(input1, weight1, bias1)
threeLayerNeuralNetwork.print_result()
threeLayerNeuralNetwork.input_to_layer_by_sigmoid()
threeLayerNeuralNetwork.print_result()

weight2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
bias2	= np.array([0.1, 0.2])

threeLayerNeuralNetwork.set_weight_and_bias(weight2, bias2)
threeLayerNeuralNetwork.input_to_layer_by_sigmoid()
threeLayerNeuralNetwork.print_result()

weight3 = np.array([[0.1, 0.3], [0.2, 0.4]])
bias3	= np.array([0.1, 0.2])

threeLayerNeuralNetwork.set_weight_and_bias(weight3, bias3)
threeLayerNeuralNetwork.input_to_layer_by_identify()
threeLayerNeuralNetwork.print_result()
