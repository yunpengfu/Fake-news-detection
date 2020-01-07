import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
import csv

if __name__ == '__main__':
    # df = pd.read_table('task1/train.csv', sep=',')
    # print('用read_table读取的csv文件：', df)
    df = pd.read_csv('task1/train.csv')
    # print('用read_csv读取的csv文件：', df)
    # print(df.text)

    # set y
    y = df.label
    # 删除label列
    df.drop("label", axis=1)
    # print(df)
    # make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df.text, y, test_size=0.2, random_state=11)
    # print("df.text.shape:"+str(df.text.shape))  # df.text.shape:(38471,)
    # print("X_train.shape:"+str(X_train.shape))  # X_train.shape:(26929,)
    # print("X_test.shape:" + str(X_test.shape))  # X_test.shape:(11542,)

    # 初始化 count_vectorizer
    # count_vectorizer = CountVectorizer(stop_words=None)
    count_vectorizer = CountVectorizer(stop_words='english')
    # fit and transform the training the training data
    count_train = count_vectorizer.fit_transform(X_train)
    # transform the test set
    count_test = count_vectorizer.transform(X_test)

    # 初始化tfidf_vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    # fit and transform the training the training data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    # df_tfidf_train = pd.DataFrame(tfidf_train)
    # df_tfidf_train.to_csv('tfidf_train.csv')
    # transform the test set
    tfidf_test = tfidf_vectorizer.transform(X_test)
    print(tfidf_test[0])
    # df_tfidf_test = pd.DataFrame(tfidf_test)
    # df_tfidf_test.to_csv('tfidf_test.csv')

    # 获得tfidf_vectorizer的特证名
    print(tfidf_vectorizer.get_feature_names()[-10:])
    # print(tfidf_test)
    # # 获得count_vectorizer的特征名
    # print(count_vectorizer.get_feature_names()[:10])

    # def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap = plt.cm.Blues):
    #
    #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #     plt.title(title)
    #     plt.colorbar()
    #     tick_marks = np.arange(len(classes))
    #     plt.xticks(tick_marks, classes, rotation=45)
    #     plt.yticks(tick_marks, classes)
    #
    #     if normalize:
    #         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #         print("Normalized confusion matrix")
    #     else:
    #         print('Confusion matrix, without normalization')
    #
    #     thresh = cm.max() / 2.
    #     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #         plt.text(j, i, cm[i, j],
    #                  horizontalalignment="center",
    #                  color="white" if cm[i, j] > thresh else "black")
    #
    #     plt.tight_layout()
    #     plt.ylabel('True label')
    #     plt.xlabel('Predicted label')
    #
    # 比较模型
    clf = MultinomialNB()
    clf.fit(tfidf_train, y_train)
    pred = clf.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    print("tfidf_train accuracy:   %0.3f" % score)
    # cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    # plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

    clf.fit(count_train, y_train)
    pred = clf.predict(count_test)
    score = metrics.accuracy_score(y_test, pred)
    print("count_train accuracy:   %0.3f" % score)
    # cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    # plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

    # tfidf_train线性模型
    linear_clf = PassiveAggressiveClassifier()
    linear_clf.fit(tfidf_train, y_train)
    pred = linear_clf.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    print("tfidf_train线性模型accuracy:   %0.3f" % score)
    # cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    # plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

    # # count_train线性模型
    # linear_clf = PassiveAggressiveClassifier(n_iter=50)
    # linear_clf.fit(count_train, y_train)
    # pred = linear_clf.predict(count_test)
    # score = metrics.accuracy_score(y_test, pred)
    # print("count_train线性模型accuracy:   %0.3f" % score)
    # # cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    # # plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

    # clf = MultinomialNB(alpha=0.1)
    # last_score = 0
    # for alpha in np.arange(0, 1, .1):
    #     nb_classifier = MultinomialNB(alpha=alpha)
    #     nb_classifier.fit(tfidf_train, y_train)
    #     pred = nb_classifier.predict(tfidf_test)
    #     score = metrics.accuracy_score(y_test, pred)
    #     if score > last_score:
    #         clf = nb_classifier
    #     print("Alpha: {:.2f} Score: {:.5f}".format(alpha, score))

    # tfidf
    df = pd.read_csv('task1/test_stage1.csv')
    # transform the test_stage1 set
    tfidf_test_stage1 = tfidf_vectorizer.transform(df.text)
    pred = linear_clf.predict(tfidf_test_stage1)
    # 写入result.csv
    # print('df.id.shape:'+str(df.id.shape)+'  pred.shape:'+str(pred.shape))
    c = np.vstack((df.id, pred))
    # print(c.T)
    df_result = pd.DataFrame(c.T, columns=["id", "label"])
    df_result.to_csv('result.csv', index=None)

    # # count
    # df = pd.read_csv('task1/test_stage1.csv')
    # # transform the test_stage1 set
    # count_test_stage1 = count_vectorizer.transform(df.text)
    # pred = linear_clf.predict(count_test_stage1)
    # # 写入result.csv
    # # print('df.id.shape:'+str(df.id.shape)+'  pred.shape:'+str(pred.shape))
    # c = np.vstack((df.id, pred))
    # # print(c.T)
    # df_result = pd.DataFrame(c.T, columns=["id", "label"])
    # df_result.to_csv('result.csv', index=None)
