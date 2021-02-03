from sklearn.model_selection import train_test_split
import pandas as pd
import config as cfg
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('agg')


def save_csv_in_memory(df, filename):

    csv = df.to_csv(index=False)

    with open(filename, "w") as file:
        file.write(csv)

    return filename


def split_data(filename, label, test_size):

    full_data = pd.read_csv(filename)

    label_local = full_data.columns.get_loc(label)

    feature_df = full_data.drop(label, axis=1)

    features = feature_df.iloc[ :, 1:]

    label_df = full_data.iloc[ :, label_local]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        label_df,
        test_size=test_size,
        random_state=0
        )

    save_csv_in_memory(x_train, 'data/x_train.csv')
    save_csv_in_memory(x_test, 'data/x_test.csv')
    save_csv_in_memory(y_train, 'data/y_train.csv')
    save_csv_in_memory(y_test, 'data/y_test.csv')

    return True


def get_accuracies(label_name, neighbs, trees, max_iter):

    x_train = pd.read_csv('data/x_train.csv')
    y_train = pd.read_csv('data/y_train.csv')
    x_test = pd.read_csv('data/x_test.csv')
    y_test = pd.read_csv('data/y_test.csv')

    y_train = np.array(y_train[label_name].to_list())
    y_test = np.array(y_test[label_name].to_list())

    accuracies = []

    accuracies.append(
        logistic_regression(
            x_train,
            y_train,
            x_test,
            y_test,
            max_iter
        )
    )

    accuracies.append(
        knn(
            x_train,
            y_train,
            x_test,
            y_test,
            neighbs
        )
    )

    accuracies.append(
        naivebayes(
            x_train,
            y_train,
            x_test,
            y_test
        )
    )

    accuracies.append(
        random_forest(
            x_train,
            y_train,
            x_test,
            y_test,
            trees
        )
    )

    df = pd.DataFrame(accuracies)
    df = df.sort_values(by=['Accuracy', 'ML Model'], ascending=[False, True])

    return df, y_train


def logistic_regression(x_train, y_train, x_test, y_test, max_iter):

    logreg = LogisticRegression(max_iter = max_iter)
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    accuracy_LR = np.sum(y_pred == y_test) / len(y_test)
    accuracy_LR
    output = {'ML Model':'Logistic Regression', 'Accuracy': accuracy_LR}

    print('Logistic Regression - done')
    return output

def knn(x_train, y_train, x_test, y_test, neighbs):

    scaler = StandardScaler()
    scaler.fit(x_train)
    X_scaled = scaler.transform(x_train)
    X2_scaled = scaler.transform(x_test)

    k_range = range(neighbs, 30)

    score_list = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k, metric = 'euclidean')
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        score_list.append(
            {
                'k': k,
                'accuracy': metrics.accuracy_score(y_test, y_pred)
            }
        )

    score_df = pd.DataFrame(score_list).sort_values(by=['accuracy', 'k'], ascending=[False, True])

    best = score_df['k'].to_list()[0]

    print('Best KNN Neighbors: ' + str(best))

    knn = KNeighborsClassifier(n_neighbors = best, metric = 'euclidean')
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy_KNN = metrics.accuracy_score(y_test, y_pred)
    output = {'ML Model':'KNN', 'Accuracy': accuracy_KNN}

    print('KNN - done')
    return output


def naivebayes(x_train, y_train, x_test, y_test):

    model = GaussianNB()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy_NB = metrics.accuracy_score(y_test, y_pred)
    output = {'ML Model':'Naive-Bayes', 'Accuracy': accuracy_NB}

    print('Naive-Bayes - done')

    return output


def random_forest(x_train, y_train, x_test, y_test, trees):

    clf = RandomForestClassifier(n_estimators=trees)

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy_RF = metrics.accuracy_score(y_test, y_pred)
    output = {'ML Model':'Random Forest', 'Accuracy': accuracy_RF}

    print('Random Forest - done')

    return output


def create_features(df_name, label_name):

    lister = []

    for col in df_name.columns:
        if col != label_name:
            lister.append(df_name[col].to_list())

    features = np.column_stack((lister))

    return features


def convert_categoricals(df):

    labelendoder_x = LabelEncoder()

    categoricals = []

    for col in df.columns:
        if df[col].dtypes not in cfg.acceptable_dtypes:
            df[col] = labelendoder_x.fit_transform(df[col])

    return df


def feature_ranking(label):

    top3_features = []

    x_train = pd.read_csv('data/x_train.csv')

    for i in range(1, 4):
        list_type = []
        sel_f = SelectKBest(f_classif, k=i)
        x_train_f = sel_f.fit_transform(x_train, label)

        for item in sel_f.get_support():
            list_type.append(str(item))

        for i in range(0, len(list_type)):
            if list_type[i] == 'True' and i not in top3_features:
                top3_features.append(i)

    top3feature_names = []

    for i in range(0, len(top3_features)):
        top3feature_names.append(
            {
                'Rank': i + 1,
                'Feature': x_train.columns[i]
            }
        )

    output = pd.DataFrame(top3feature_names)

    return output


def plot_accuracies(filename):

    df = pd.read_csv(filename)

    #plt.figure(figsize=(12, 8))
    ax = df.plot.bar(x='ML Model', y='Accuracy')
    ax.set_title('Model Accuracy')
    plt.tight_layout()

    plt.savefig('./static/accuracies.png', bbox_inches='tight', pad_inches = 0.0)

    return True


def plot_confusion_matrix(y_test, y_pred):

    cnf_matrix_NB = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix_NB

    fig, ax = plt.subplots()

    sns.heatmap(cnf_matrix_NB, annot=True, cmap="PuBuGn" ,fmt='d')
    ax.xaxis.set_label_position("bottom")
    plt.tight_layout()
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.savefig('./static/confusion.png', bbox_inches='tight', pad_inches = 0.0)

    return True


def label_test(label_pick, filename):
    label_test = pd.read_csv(filename)
    label_vals = label_test[label_pick].to_list()

    label_check = []

    for label in label_vals:
        if label not in label_check:
            label_check.append(label)

    if len(label_check) < 3:
        return 1, label_test.columns
    else:
        return 0, label_test.columns
