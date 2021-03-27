from PIL import Image
from sklearn.model_selection import  train_test_split
from feature import NPDFeature
from tqdm import tqdm
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

def get_i_face_image_path(i):
    path = './datasets/original/face/face_'
    if i < 10:
        path = path + '00' + str(i)
    else:
        if i < 100:
            path = path + '0' + str(i)
        else:
            path = path + str(i)
    path = path + '.jpg'
    return path

def get_i_nonface_image_path(i):
    path = './datasets/original/nonface/nonface_'
    if i < 10:
        path = path + '00' + str(i)
    else:
        if i < 100:
            path = path + '0' + str(i)
        else:
            path = path + str(i)
    path = path + '.jpg'
    return path

def read_data():
    X = []
    y = []
    for i in range(500):
        path = get_i_face_image_path(i)
        image = Image.open(path)
        image = image.convert('L')
        image = image.resize((16, 16))
        X.append(image)
        y.append(1)
    for i in range(500):
        path = get_i_nonface_image_path(i)
        image = Image.open(path)
        image = image.convert('L')
        image = image.resize((16, 16))
        X.append(image)
        y.append(-1)

    feature = []
    for i in tqdm(range(len(X)), desc='pre_train', leave=True):
        array_image = np.array(X[i])
        fea = NPDFeature(array_image)
        feature.append(fea.extract())
    return feature, y


if __name__ == "__main__":
    # write your code here
    X, y = read_data()
    with open("data.pickle", "wb") as f:
        pickle.dump((X, y), f, pickle.HIGHEST_PROTOCOL)
    with open("data.pickle", "rb") as f:
        X, y = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

    classifiler = AdaBoostClassifier(DecisionTreeClassifier, 7)
    classifiler.fit(X_train, y_train)

    pred_y_test = classifiler.predict(X_test)
    report = classification_report(y_test, pred_y_test, labels=[-1, 1], target_names=['face', 'nonface'])
    print(report)
    '''
    pred_test_error_list = []
    pred_test_error_list.append(None)
    pred_train_error_list = []
    pred_train_error_list.append(None)
    for iter_s in tqdm(range(1, 20), desc='test', leave=True):
        classifiler = AdaBoostClassifier(DecisionTreeClassifier, iter_s)
        classifiler.fit(X_train, y_train)
        pred_y_train = classifiler.predict(X_train)
        error_rate_count_train = 0
        for i in range(len(y_train)):
            if pred_y_train[i] != y_train[i]:
                error_rate_count_train += 1
        error_rate_count_train /= len(y_train)
        #print('error rate of train: ' + str(error_rate_count_train))

        pred_y_test = classifiler.predict(X_test)
        error_rate_count_test = 0
        for i in range(len(y_test)):
            if pred_y_test[i] != y_test[i]:
                error_rate_count_test += 1
        error_rate_count_test /= len(y_test)
        #print('error rate of test: ' + str(error_rate_count_test))
        pred_test_error_list.append(error_rate_count_test)
        pred_train_error_list.append(error_rate_count_train)
    plt.figure()
    plt.plot(pred_train_error_list, label='train error rate', color='blue')
    plt.plot(pred_test_error_list, label='test error rate', color='yellow')
    plt.xlabel('weak classifiler number')
    plt.ylabel('error rate')
    plt.title('classifiler number-error rate')
    plt.legend()
    plt.show()
    plt.savefig('number_error.jpg')
    '''
    with open('classifiler_result.txt', 'w') as result:
        result.write(report)
