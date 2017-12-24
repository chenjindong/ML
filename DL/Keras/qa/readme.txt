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

2.用多分类问题建模
	数据集介绍：
		11600个样本，420个类（人物、地点、演员...）
	method1：
		深度模型，validation set和test set上acc在52%左右，具体数据见valid_predict.txt和test_predict.txt
		


	