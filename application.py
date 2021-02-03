from flask import Flask, render_template, request, Response, send_file, after_this_request
import pandas as pd
import requests
import json
import config as cfg
import numpy as np
import helper_functions as hlp
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pathlib

application = app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def landing():
    if request.method != 'POST':
        base_path = pathlib.Path('./static/')
        for f_name in base_path.iterdir():
            if str(f_name) != 'static/main.css':
                os.remove(f_name)
        base_path = pathlib.Path('./data/')
        for f_name in base_path.iterdir():
            if str(f_name) != 'data/dummy_data.csv':
                os.remove(f_name)
        return render_template('landing.html')
    else:
        return render_template('landing.html')


@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
        filename = 'data/full_dataset.csv'
        cleaned_df = hlp.convert_categoricals(df)
        saved = hlp.save_csv_in_memory(cleaned_df, filename)
        columns = []
        for col in cleaned_df.columns:
            columns.append(col)
        return render_template(
            'selections.html',
            columns=columns,
            sizes=cfg.split_options
        )

    return render_template('home.html')


@app.route('/selections', methods=['GET', 'POST'])
def select_label():
    if request.method == 'POST':
        label_pick = request.form.get("label_pick", None)
        split_pick = int(request.form.get("split_pick", None))
        neighbs = int(request.form.get("neighbs", None))
        trees = int(request.form.get("trees", None))
        max_iter = int(request.form.get("max_iter", None))

        label_check, features = hlp.label_test(label_pick, 'data/full_dataset.csv')

        if label_check == 1:

            test_size = (100 - split_pick) / 100

            hlp.split_data('data/full_dataset.csv', label_pick, test_size)
            output_df, y_train = hlp.get_accuracies(label_pick, neighbs, trees, max_iter)

            hlp.save_csv_in_memory(
                output_df,
                'data/accuracies.csv'
            )

            hlp.save_csv_in_memory(
                hlp.feature_ranking(y_train),
                'data/features.csv'
            )

            hlp.plot_accuracies('data/accuracies.csv')

            features = pd.read_csv('data/features.csv')
            features = features['Feature'].to_list()

            for file in cfg.file_list:
                os.remove(file)

            os.remove('data/accuracies.csv')
            os.remove('data/features.csv')
            #os.remove('static/accuracies.png')

            return render_template(
                'model_specs.html',
                url ='/static/accuracies.png',
                feat1 = cfg.feat1 + str(features[0]),
                feat2 = cfg.feat2 + str(features[1]),
                feat3 = cfg.feat3 + str(features[2]),
                )

        else:
            return render_template(
                'selections.html',
                columns=features,
                sizes=cfg.split_options,
                error=cfg.label_error,
            )

    return render_template(
        'selections.html',
        columns=columns,
        sizes=cfg.split_options
    )


@app.route('/model_specs', methods=['GET', 'POST'])
def model_specs():
    return render_template(
        'model_specs.html'
        )



@app.route('/model_chosen', methods=['GET', 'POST'])
def model_chosen():
    label_pick = request.form.get("label_pick", None)
    split_pick = int(request.form.get("split_pick", None))

    test_size = (100 - split_pick) / 100

    label_check, features = hlp.label_test(label_pick, 'data/full_dataset.csv')

    if label_check == 1:

        hlp.split_data('data/full_dataset.csv', label_pick, test_size)

        model_pick = request.form.get("tests", None)

        x_train = pd.read_csv('data/x_train.csv')
        y_train = pd.read_csv('data/y_train.csv')
        x_test = pd.read_csv('data/x_test.csv')
        y_test = pd.read_csv('data/y_test.csv')

        if model_pick == cfg.tests[0]:
            return render_template(
                'knn_single.html',
                label=label_pick
            )
        elif model_pick == cfg.tests[1]:
            return render_template(
                'logistic_single.html',
                label=label_pick
            )


        elif model_pick == cfg.tests[2]:
            model = GaussianNB()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            hlp.plot_confusion_matrix(y_test, y_pred)

            overall_accuracy = metrics.accuracy_score(y_test, y_pred) * 100
            overall_accuracy = str(overall_accuracy)[:5]

            model_name = 'Naive-Bayes'

            hlp.save_csv_in_memory(
                hlp.feature_ranking(y_train),
                'data/features.csv'
            )

            features = pd.read_csv('data/features.csv')
            features = features['Feature'].to_list()

            return render_template(
                'single_test.html',
                url ='/static/confusion.png',
                feat1 = cfg.feat1 + str(features[0]),
                feat2 = cfg.feat2 + str(features[1]),
                feat3 = cfg.feat3 + str(features[2]),
                model_name = model_name,
                overall_accuracy = overall_accuracy
            )
        else:
            return render_template(
                'random_single.html',
                label=label_pick
            )

    else:

        return render_template(
            'model_select.html',
            error=cfg.label_error,
            columns=features,
            sizes=cfg.split_options,
            tests=cfg.tests
        )


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
        filename = 'data/full_dataset.csv'
        cleaned_df = hlp.convert_categoricals(df)
        saved = hlp.save_csv_in_memory(cleaned_df, filename)
        columns = []
        for col in cleaned_df.columns:
            columns.append(col)
        return render_template(
            'model_select.html',
            tests=cfg.tests,
            columns=columns,
            sizes=cfg.split_options
        )
    return render_template(
        'home_single.html'
    )


@app.route('/dummy_selections', methods=['GET', 'POST'])
def dummy_selections():
    if request.method == 'POST':
        label_pick = request.form.get("label_pick", None)
        split_pick = int(request.form.get("split_pick", None))
        neighbs = int(request.form.get("neighbs", None))
        trees = int(request.form.get("trees", None))
        max_iter = int(request.form.get("max_iter", None))

        label_check, features = hlp.label_test(label_pick, 'data/dummy_data.csv')

        if label_check == 1:

            test_size = (100 - split_pick) / 100

            df = pd.read_csv('data/dummy_data.csv')

            cleaned_df = hlp.convert_categoricals(df)
            hlp.save_csv_in_memory(cleaned_df, 'data/dummy_data.csv')

            hlp.split_data('data/dummy_data.csv', label_pick, test_size)
            output_df, y_train = hlp.get_accuracies(label_pick, neighbs, trees, max_iter)

            hlp.save_csv_in_memory(
                output_df,
                'data/accuracies.csv'
            )

            hlp.save_csv_in_memory(
                hlp.feature_ranking(y_train),
                'data/features.csv'
            )

            hlp.plot_accuracies('data/accuracies.csv')

            features = pd.read_csv('data/features.csv')
            features = features['Feature'].to_list()

            for file in cfg.file_list:
                if file != 'data/full_dataset.csv':
                    os.remove(file)

            os.remove('data/accuracies.csv')
            os.remove('data/features.csv')
            #os.remove('static/accuracies.png')

            return render_template(
                'model_specs.html',
                url ='/static/accuracies.png',
                feat1 = cfg.feat1 + str(features[0]),
                feat2 = cfg.feat2 + str(features[1]),
                feat3 = cfg.feat3 + str(features[2]),
                )

        else:
            return render_template(
                'dummy_selections.html',
                columns=features,
                sizes=cfg.split_options,
                error=cfg.label_error,
            )
    dummy_data = pd.read_csv('data/dummy_data.csv')
    columns = dummy_data.columns
    return render_template(
        'dummy_selections.html',
        columns=columns,
        sizes=cfg.split_options
    )


@app.route('/logistic', methods=['GET', 'POST'])
def logistic():

    max_iter = int(request.form.get("max_iter", None))

    x_train = pd.read_csv('data/x_train.csv')
    y_train = pd.read_csv('data/y_train.csv')
    x_test = pd.read_csv('data/x_test.csv')
    y_test = pd.read_csv('data/y_test.csv')

    logreg = LogisticRegression(max_iter = max_iter)
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    hlp.plot_confusion_matrix(y_test, y_pred)

    overall_accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    overall_accuracy = str(overall_accuracy)[:5]

    model_name = 'Logistic Regression'

    hlp.save_csv_in_memory(
        hlp.feature_ranking(y_train),
        'data/features.csv'
    )

    features = pd.read_csv('data/features.csv')
    features = features['Feature'].to_list()

    return render_template(
        'single_test.html',
        url ='/static/confusion.png',
        feat1 = cfg.feat1 + str(features[0]),
        feat2 = cfg.feat2 + str(features[1]),
        feat3 = cfg.feat3 + str(features[2]),
        model_name = model_name,
        overall_accuracy = overall_accuracy
    )


@app.route('/knn', methods=['GET', 'POST'])
def knn():
    neighbs = int(request.form.get("neighbs", None))

    x_train = pd.read_csv('data/x_train.csv')
    y_train = pd.read_csv('data/y_train.csv')
    x_test = pd.read_csv('data/x_test.csv')
    y_test = pd.read_csv('data/y_test.csv')

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

    knn = KNeighborsClassifier(n_neighbors = best, metric = 'euclidean')
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    hlp.plot_confusion_matrix(y_test, y_pred)

    overall_accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    overall_accuracy = str(overall_accuracy)[:5]

    model_name = 'K Nearest Neighbor'

    hlp.save_csv_in_memory(
        hlp.feature_ranking(y_train),
        'data/features.csv'
    )

    features = pd.read_csv('data/features.csv')
    features = features['Feature'].to_list()

    return render_template(
        'single_test.html',
        url ='/static/confusion.png',
        feat1 = cfg.feat1 + str(features[0]),
        feat2 = cfg.feat2 + str(features[1]),
        feat3 = cfg.feat3 + str(features[2]),
        model_name = model_name,
        overall_accuracy = overall_accuracy
    )


@app.route('/random', methods=['GET', 'POST'])
def random():
    trees = int(request.form.get("trees", None))

    x_train = pd.read_csv('data/x_train.csv')
    y_train = pd.read_csv('data/y_train.csv')
    x_test = pd.read_csv('data/x_test.csv')
    y_test = pd.read_csv('data/y_test.csv')

    clf = RandomForestClassifier(n_estimators=trees)

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    hlp.plot_confusion_matrix(y_test, y_pred)

    overall_accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    overall_accuracy = str(overall_accuracy)[:5]

    model_name = 'Random Forest Classifier'

    hlp.save_csv_in_memory(
        hlp.feature_ranking(y_train),
        'data/features.csv'
    )

    features = pd.read_csv('data/features.csv')
    features = features['Feature'].to_list()

    return render_template(
        'single_test.html',
        url ='/static/confusion.png',
        feat1 = cfg.feat1 + str(features[0]),
        feat2 = cfg.feat2 + str(features[1]),
        feat3 = cfg.feat3 + str(features[2]),
        model_name = model_name,
        overall_accuracy = overall_accuracy
    )


@app.route('/dummy_single', methods=['GET', 'POST'])
def dummy_single():
    dummy_data = pd.read_csv('data/dummy_data.csv')
    columns = dummy_data.columns

    if request.method == 'POST':
        label_pick = request.form.get("label_pick", None)
        split_pick = int(request.form.get("split_pick", None))
        test_size = (100 - split_pick) / 100

        hlp.split_data('data/dummy_data.csv', label_pick, test_size)

        model_pick = request.form.get("tests", None)

        x_train = pd.read_csv('data/x_train.csv')
        y_train = pd.read_csv('data/y_train.csv')
        x_test = pd.read_csv('data/x_test.csv')
        y_test = pd.read_csv('data/y_test.csv')

        if model_pick == cfg.tests[0]:
            return render_template(
                'knn_single.html',
                label=label_pick
            )
        elif model_pick == cfg.tests[1]:
            return render_template(
                'logistic_single.html',
                label=label_pick
            )


        elif model_pick == cfg.tests[2]:
            model = GaussianNB()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            hlp.plot_confusion_matrix(y_test, y_pred)

            overall_accuracy = metrics.accuracy_score(y_test, y_pred) * 100
            overall_accuracy = str(overall_accuracy)[:5]

            model_name = 'Naive-Bayes'

            hlp.save_csv_in_memory(
                hlp.feature_ranking(y_train),
                'data/features.csv'
            )

            features = pd.read_csv('data/features.csv')
            features = features['Feature'].to_list()

            return render_template(
                'single_test.html',
                url ='/static/confusion.png',
                feat1 = cfg.feat1 + str(features[0]),
                feat2 = cfg.feat2 + str(features[1]),
                feat3 = cfg.feat3 + str(features[2]),
                model_name = model_name,
                overall_accuracy = overall_accuracy
            )
        else:
            return render_template(
                'random_single.html',
                label=label_pick
            )

    else:
        return render_template(
            'dummy_single.html',
            tests=cfg.tests,
            columns=columns,
            sizes=cfg.split_options
        )


if __name__ == "__main__":
    app.run(host ='0.0.0.0', port = 80, debug = True)
