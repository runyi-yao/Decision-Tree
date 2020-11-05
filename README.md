## - 利用ID3决策树预测患糖尿病的可能性
关于患糖尿病可能性的预测
  1.主要实验流程
  获取数据集—->创建 ID3 决策树--->绘制决策树--->模型测试
  if __name__ == '__main__':
   # 获取数据集
   dataSet, labels = getDataSet()
   featLabels = []
   # 创建 ID3 决策树
   myTree = createTree(dataSet, labels, featLabels)
   # 绘制决策树
   createPlot(myTree)
   # 模型测试
   modelTest(myTree, featLabels)
