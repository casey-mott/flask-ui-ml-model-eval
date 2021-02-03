# flask-ui-ml-model-eval
Flask web UI for evaluating machine learning models

# Preface
Machine Learning classification is unnecessarily complex for basic applications or use cases. This application attempts to provide a sample a non code solution to implementing the basic ML classification models:

1. Logistic Regression
2. Naive Bayes
3. K Nearest Neighbor
4. Random Forest

# Requirements

- Basic understanding of the ML models (for input variables)
- Cleaned, pre-processed data to train and test
- Basic understanding of the Terminal

# How to Host the Application

Clone the repository locally. 

In the Terminal, navigate to the cloned repo location and run the following commands: 

If virtual env is not installed:

```
pip install virtualenv
```

Start virtual environment:

```
virtualenv virt
```

```
source virt/bin/activate
```

Install requirements: 
```
pip install -r requirements.txt
```

Launch application: 
```
python application.py
```

Once application has been launched, it will be hosted locally on your machine, accessible here. (Performance is best in Chrome)
```
http://0.0.0.0:80/
```

# Using the Application

The applications accepts cleaned CSV uploads of any data sets for training and testing. The data that is uploaded must first be pre-processed. 

Alternatively, there is dummy data stored in the application's memory, which is sourced from Kaggle. It is a dataset using HR analytics to classify employees likely to leave a company. 

The user has the choice to select a single model to see overall performance or can compare all of the models to find the best performing classifier. 
