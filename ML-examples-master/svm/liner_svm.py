from sklearn import svm

x = [[2, 0, 1], [1, 1, 2], [2, 3, 3]]
y = [0, 0, 1]  # 分类标记
clf = svm.SVC(kernel='linear')  # SVM模块，svc,线性核函数
clf.fit(x, y)

print(clf)

print(clf.support_vectors_)  # 支持向量点

print(clf.support_)  # 支持向量点的索引

print(clf.n_support_)  # 每个class有几个支持向量点

print(clf.predict([[2, 0, 3]]))  # 预测