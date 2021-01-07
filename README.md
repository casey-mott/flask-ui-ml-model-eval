# flask-ui-ml-model-eval
Flask web UI for evaluating machine learning models

# Preface

Flask is a python web framework that can be used to layer a UI on top of python code, using HTML and JavaScript to define interface templates and routing between subdomains.

Machine learning use cases often involve the evaluation of different models to find the most accurate results. Accuracy is defined by the percentage of correct labels in a test data set on a trained model. 

This UI seeks to: 
1. Take in a data set with labels;
2. Split the data set into training and testing data;
3. Learn data types and preprocess data for evaluation;
4. Run training data through selected models;
5. Evaluate accuracy of each selected model based on test data set; and,
6. Visualize the results.

Due to the nature of machine learning models, there are inputs that are required from the user. For example, in the Random Forest classifier one must define the number of decision trees to use. 
