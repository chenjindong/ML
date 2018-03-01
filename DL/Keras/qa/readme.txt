Background:
	现在KBQA有一个非常典型的问题——“答非所问”。e.g.
		question：姚明的身高多少？
		answer：上海
	很明显这是一个错误的答案，如果我们已经知道了answer type，那么就不会出现那么离谱的错误回答。
Task：
	input：question
	output：answer type
	example：
		input：谁演唱了青花瓷？
    		output：人物

Solution：
1. 构建训练集T，格式为（question，answer type）
	subtask：
		已有数据question和answer，比如（谁演唱了青花瓷？，周杰伦）
		需要构建训练集question和answer type，比如（谁演唱了青花瓷？，人物）
	method：
		利用中文分类体系（CN-Probase）对answer进行泛化，得到训练集T

2.1 用多分类问题建模
	description：
		数据集：11600个样本，420个类（人物、地点、演员...）
	method1：
		深度模型
	result：
		validation set accuracy：0.54
		test set accuracy：0.59
		预测数据见valid_predict.txt和test_predict.txt
	Tip：本质上是多标签多分类问题，比如周杰伦，预测他是人物和演员都可以，故实际的accuracy大于60以上
2.2 构建N个分类器
	description：
		为每个answer type构建一个二分类器，比如“人物”对应的label为（人物，非人物）
	
	


	
