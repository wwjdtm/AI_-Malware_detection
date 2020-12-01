import csv
import datetime
import glob  # noqa
import json
import os  # noqa
import pprint  # noqa
import time

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline  # noqa
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 41


def read_label_csv(_path):
    label_table = dict()
    path = './데이터/' + _path
    with open(path, "r") as f:
        for line in f.readlines()[1:]:
            fname, label = line.strip().split(",")
            label_table[fname] = int(label)
    return label_table


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_model(**kwargs):
    if kwargs["model"] == "rf":
        return RandomForestClassifier(random_state=kwargs["random_state"], n_jobs=4)
    elif kwargs["model"] == "dt":
        return DecisionTreeClassifier(random_state=kwargs["random_state"])
    elif kwargs["model"] == "lgb":
        return LGBMClassifier(random_state=kwargs["random_state"])
    elif kwargs["model"] == "svm":
        return SVC(random_state=kwargs["random_state"])
    elif kwargs["model"] == "lr":
        return LogisticRegression(random_state=kwargs["random_state"], n_jobs=-1)
    elif kwargs["model"] == "knn":
        return KNeighborsClassifier(n_jobs=-1)
    elif kwargs["model"] == "adaboost":
        return AdaBoostClassifier(random_state=kwargs["random_state"])
    elif kwargs["model"] == "mlp":
        return MLPClassifier(random_state=kwargs["random_state"])
    else:
        print("Unsupported Algorithm")
        return None


def train(X_train, y_train, model):
    '''
        머신러닝 모델을 선택하여 학습을 진행하는 함수
        :param X_train: 학습할 2차원 리스트 특징벡터
        :param y_train: 학습할 1차원 리스트 레이블 벡터
        :param model: 문자열, 선택할 머신러닝 알고리즘
        :return: 학습된 머신러닝 모델 객체
    '''
    clf = load_model(model=model, random_state=SEED)
    clf.fit(X_train, y_train)
    return clf


def evaluate(X_test, y_test, model):
    '''
        학습된 머신러닝 모델로 검증 데이터를 검증하는 함수
        :param X_test: 검증할 2차원 리스트 특징 벡터
        :param y_test: 검증할 1차원 리스트 레이블 벡터
        :param model: 학습된 머신러닝 모델 객체
    '''
    predict = model.predict(X_test)
    print(f"{model} 정확도", model.score(X_test, y_test))


class PeminerParser:
    def __init__(self, path):
        self.report = read_json(path)
        self.vector = []

    def process_report(self):
        '''
            전체 데이터 사용
        '''

        self.vector = [value for _, value in sorted(self.report.items(), key=lambda x: x[0])]
        return self.vector


class EmberParser:
    '''
        예제에서 사용하지 않은 특징도 사용하여 벡터화 할 것을 권장
    '''

    def __init__(self, path):
        self.report = read_json(path)
        self.vector = []

    def get_histogram_info(self):
        histogram = np.array(self.report["histogram"])
        total = histogram.sum()
        vector = histogram / total
        return vector.tolist()

    def get_byteentropy(self):  # byteentropy 특징 벡터 추가
        byteentropys = np.array(self.report["byteentropy"])
        vector = byteentropys
        return vector.tolist()

    def get_string_info(self):
        strings = self.report["strings"]

        hist_divisor = float(strings['printables']) if strings['printables'] > 0 else 1.0
        vector = [
            strings['numstrings'],
            strings['avlength'],
            strings['printables'],
            strings['entropy'],
            strings['paths'],
            strings['urls'],
            strings['registry'],
            strings['MZ']
        ]
        vector += (np.asarray(strings['printabledist']) / hist_divisor).tolist()
        return vector

    def get_datadirectories(self):  # datadirectories size 특징 추가
        # {'name': 'EXPORT_TABLE',
        #                   'size': 6151,
        #                   'virtual_address': 1032272},15개
        datadirectories = self.report["datadirectories"]

        vector = []

        if len(datadirectories) != 15:
            vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            print("dhdhdh")
        else:
            for a in range(len(datadirectories)):
                size = datadirectories[a]["size"] if datadirectories[a]["size"] > 100 else 0
                virtual_address = datadirectories[a]["virtual_address"] if datadirectories[a][
                                                                               "virtual_address"] > 100000 else 0
                # vector+= datadirectories[a]["size"], datadirectories[a]["virtual_address"]
                vector += size, virtual_address

        return vector

    def get_general_file_info(self):
        general = self.report["general"]
        vector = [
            general['size'], general['vsize'], general['has_debug'], general['exports'], general['imports'],
            general['has_relocations'], general['has_resources'], general['has_signature'], general['has_tls'],
            general['symbols']
        ]
        return vector

    def get_header_info(self):
        header = self.report['header']
        vector = [
            header['coff']['timestamp'],
            # header['coff']['machine'],
            # header['coff']['characteristics'],
            # header['optional']['subsystem'],
            # header['optional']['dll_characteristics'],
            # header['optional']['magic'],
            header['optional']['major_image_version'],
            header['optional']['minor_image_version'],
            header['optional']['major_linker_version'],
            header['optional']['minor_linker_version'],
            header['optional']['major_operating_system_version'],
            header['optional']['minor_operating_system_version'],
            header['optional']['major_subsystem_version'],
            header['optional']['minor_subsystem_version'],
            header['optional']['sizeof_code'],
            header['optional']['sizeof_headers'],
            header['optional']['sizeof_heap_commit'],
        ]
        return vector

    def get_section_info(self):
        section = self.report['section']
        vector = []
        for i in range(50):
            vector.append(0)
        for index, value in enumerate(section['sections']):
            vector[index] = value['size']
            vector[index] = value['entropy']
            vector[index] = value['vsize']
        return vector

    def process_report(self):
        vector = []
        vector += self.get_byteentropy()
        vector += self.get_general_file_info()
        vector += self.get_histogram_info()
        vector += self.get_string_info()
        vector += self.get_datadirectories()
        vector += self.get_section_info()
        vector += self.get_header_info()

        '''
            특징 추가
        '''

        # vector += self.get_section_info()
        # vector += self.get_datadirectories_info()

        return vector


class PestudioParser:
    '''
        사용할 특징을 선택하여 벡터화 할 것을 권장
    '''

    def __init__(self, path):
        self.report = read_json(path)
        self.vector = []

    def process_report(self):
        '''
              전체 데이터 사용
        '''
        self.vector = [value for _, value in sorted(self.report.items(), key=lambda x: x[0])]
        return self.vector


def ensemble_result(X, y, models):
    '''
        학습된 모델들의 결과를 앙상블하는 함수
        :param X: 검증할 2차원 리스트 특징 벡터
        :param y: 검증할 1차원 리스트 레이블 벡터
        :param models: 1개 이상의 학습된 머신러닝 모델 객체를 가지는 1차원 리스트
    '''

    # Soft Voting
    # https://devkor.tistory.com/entry/Soft-Voting-%EA%B3%BC-Hard-Voting
    predicts = []
    for model in models:
        prob = [result for _, result in model.predict_proba(X)]
        predicts.append(prob)

    predict = np.mean(predicts, axis=0)
    predict = [1 if x >= 0.5 else 0 for x in predict]

    if os.environ['MODE'] != 'DEV':
        label_table = read_label_csv("검증데이터_정답.csv").keys()
        with open('predict.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['file', 'predict'])
            for index, value in enumerate(label_table):
                writer.writerow([value, predict[index]])

    print("앙상블 정확도", accuracy_score(y, predict))


def select_feature(X, y, model):
    '''
        주어진 특징 벡터에서 특정 알고리즘 기반 특징 선택
        본 예제에서는 RFE 알고리즘 사용
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE.fit_transform
        :param X: 검증할 2차원 리스트 특징 벡터
        :param y: 검증할 1차원 리스트 레이블 벡터
        :param model: 문자열, 특징 선택에 사용할 머신러닝 알고리즘
    '''

    model = load_model(model=model, random_state=SEED)
    rfe = RFE(estimator=model)
    return rfe.fit_transform(X, y)


def create_feature(folder='학습데이터'):
    label_table = read_label_csv(f"{folder}_정답.csv")
    X, y = [], []

    file_list = os.listdir(f'./데이터/EMBER/{folder}/')
    # print('file_list', file_list)

    for index, value in enumerate(file_list):
        file_list[index] = file_list[index].replace('.json', '')

    no_such_count = 0
    for fname in file_list:
        feature_vector = []
        try:
            label = label_table[fname]
            for data in ["PEMINER", "EMBER"]:
                path = f"./데이터/{data}/{folder}/{fname}.json"
                if data == "PEMINER":
                    feature_vector += PeminerParser(path).process_report()
                # elif data == "PESTUDIO":
                #     feature_vector += PestudioParser(path).process_report()
                else:
                    feature_vector += EmberParser(path).process_report()
            X.append(feature_vector)
            y.append(label)
        except Exception as e:
            pass
    #         print(e)
    #         no_such_count += 1
    # print(no_such_count)
    return X, y


def main():
    start_time = time.time()  # 시작 시간
    print('시작시간 : ', datetime.datetime.today())

    # label_table = read_label_csv("학습데이터_정답.csv")
    # print(label_table)

    # return
    '''
    '''
    # ember_path = "./데이터/EMBER/학습데이터/000c4ae5e00a1d4de991a9decf9ecbac59ed5582f5972f05b48bc1a1fe57338a.json"
    # peminer_path = "./데이터/PEMINER/학습데이터/000c4ae5e00a1d4de991a9decf9ecbac59ed5582f5972f05b48bc1a1fe57338a.json"
    #
    # ember_result = read_json(ember_path)

    # peminer_result = read_json(peminer_path)

    # pprint.pprint(ember_result)
    # pprint.pprint(peminer_result)
    # return

    '''
    '''
    # 데이터의 특징 벡터 모음(2차원 리스트) : X
    # 데이터의 레이블 모음(1차원 리스트) : y

    # print(np.asarray(X).shape, np.asarray(y).shape)

    # return
    '''
    '''

    X, y = create_feature('학습데이터')

    # 학습
    models = []
    for model in ["rf", "lgb"]:
        clf = train(X, y, model)
        models.append(clf)

    # 검증
    # 실제 검증 시에는 제공한 검증데이터를 검증에 사용해야 함

    eva_X, eva_y = create_feature('검증데이터')
    for model in models:
        evaluate(eva_X, eva_y, model)

    '''
    '''
    ensemble_result(eva_X, eva_y, models)

    # return
    '''
    '''

    print('걸린 시간 : ', time.time() - start_time)

    # selected_X = select_feature(X, y, "rf")
    # new_model = train(selected_X, y, "rf")


main()