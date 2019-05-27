#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import os
dest=os.path.join('SpeedClassifier','pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(clf,open(os.path.join(dest,'classifier.pkl'),'wb'),protocol=None)


# In[ ]:


import pickle
import re
import os

clf=pickle.load(open(os.path.join('pkl_objects','classifier.pkl'),'rb'))


# In[ ]:


import numpy as np
label={0:'negative',1:'positive'}


# In[ ]:


clf.predict(X)


# In[ ]:


clf.predict(X)[0]


# In[ ]:


clf.predict_proba(X)       #predict_proba() method to return the corresponding probability of our prediction.


# In[ ]:


label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100


#  predict_proba() method returns an array with a probability value for each unique class label.
# 
# Since the class label with the largest probability corresponds to the class label that is returned by the predict(), we used the np.max() to return the probability of the predicted class.

# In[ ]:


import sqlite3
import os

conn=sqlite3.connect('spped_record.sqlite')
c=conn.cursor()

c.execute('CREATE TABLE speed_db Road_Surface DOUBLE, Timelevel INTEGER, Wifi_density DOUBLE, Intersection_density DOUBLE, Honk_duration INTEGER')

c.execute("INSERT INTO speed_db Road_Surface,Timelevel,Wifi_density,Intersection_density,Honk_duration VALUES (?,?,?,?)")

c.commit()
c.close()


# In[ ]:


from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


if __name__ == '__main__':
    app.run(debug=True)


# After importing, we create an instance of the Flask class and pass in the __name__ variable that Python fills in for us. This variable will be "__main__", if this file is being directly run through Python as a script. If we import the file instead, the value of __name__ will be the name of the file where we did the import. For instance, if we had test.py and run.py, and we imported test.py into run.py the __name__ value of test.py will be test.

# In[ ]:


import pickle

with open("python_lin_reg_model.pkl", "wb") as file_handler:
    pickle.dump(lin_reg, file_handler)
    
with open("python_lin_reg_model.pkl", "rb") as file_handler:
    loaded_pickle = pickle.load(file_handler)
    


# While this method works, scikit-learn has their own model persistence method we will use: joblib. This is more efficient to use with scikit-learn models due to it being better at handling larger numpy arrays that may be stored in the models.

# In[1]:


from sklearn.externals import joblib

joblib.dump(lin_reg, "linear_regression_model.pkl")


# The prediction API is quite simple. We give it our data, the years of experience, and pass that into our predict method of our model.
# 
# 

# In[ ]:


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            years_of_experience = float(data["yearsOfExperience"])
            
            lin_reg = joblib.load("./linear_regression_model.pkl")
        except ValueError:
            return jsonify("Please enter a number.")

        return jsonify(lin_reg.predict(years_of_experience).tolist())


# In[ ]:


import requests

years_exp = {"yearsOfExperience": 8}

response = requests.post("{}/predict".format(BASE_URL), json = years_exp)

response.json()


# We’d have to retrain it with all of your old data, plus your new data.
# 
# In order to do this, we will need to save out the training data and labels.

# In[ ]:


@app.route("/retrain", methods=['POST'])
def retrain():
    if request.method == 'POST':
        data = request.get_json()

        try:
            training_set = joblib.load("./training_data.pkl")
            training_labels = joblib.load("./training_labels.pkl")

            df = pd.read_json(data)

            df_training_set = df.drop(["Salary"], axis=1)
            df_training_labels = df["Salary"]

            df_training_set = pd.concat([training_set, df_training_set])
            df_training_labels = pd.concat([training_labels, df_training_labels])

            new_lin_reg = LinearRegression()
            new_lin_reg.fit(df_training_set, df_training_labels)

            os.remove("./linear_regression_model.pkl")
            os.remove("./training_data.pkl")
            os.remove("./training_labels.pkl")

            joblib.dump(new_lin_reg, "linear_regression_model.pkl")
            joblib.dump(df_training_set, "training_data.pkl")
            joblib.dump(df_training_labels, "training_labels.pkl")

            lin_reg = joblib.load("./linear_regression_model.pkl")
        except ValueError as e:
            return jsonify("Error when retraining - {}".format(e))

        return jsonify("Retrained model successfully.")


# data = json.dumps([{"YearsExperience": 12,"Salary": 140000}, 
#                    {"YearsExperience": 12.1,"Salary": 142000}])
# With this new data, we can then call the retrain API.
# 
# response = requests.post("{}/retrain".format(BASE_URL), json = data)
# 
# response.json()
# 

# Details such as the coefficients and intercepts of the model and the current score of the model may be another useful endpoint for our API.
# 
# 

# In[ ]:


from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd

headers = ['times_pregnant', 'glucose', 'blood_pressure', 'skin_fold_thick', 'serum_insuling', 'mass_index', 'diabetes_pedigree', 'age']

with open(f'diabetes-model.pkl', 'rb') as f:
    model = pickle.load(f)
    
nput_variables = pd.DataFrame([[1, 106, 70, 28, 135, 34.2, 0.142, 22]],
                                columns=headers, 
                                dtype=float,
                                index=['input'])
# Get the model's prediction
prediction = model.predict(input_variables)
print("Prediction: ", prediction)
prediction_proba = model.predict_proba(input_variables)
print("Probabilities: ", prediction_proba)


# In[ ]:


app = Flask(__name__)
CORS(app)
@app.route("/katana-ml/api/v1.0/diabetes", methods=['POST'])
def predict():
    payload = request.json['data']
    values = [float(i) for i in payload.split(',')]
    
    input_variables = pd.DataFrame([values],
                                columns=headers, 
                                dtype=float,
                                index=['input'])
    # Get the model's prediction
    prediction_proba = model.predict_proba(input_variables)
    prediction = (prediction_proba[0])[1]
    
    ret = '{"prediction":' + str(float(prediction)) + '}'
    
    return ret
# running REST interface, port=5000 for direct test
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)


# FULL CODE TOWARDS SCIENCE

# In[ ]:


import pandas as pd
df = pd.read_csv('titanic.csv')
include = ['Age', 'Sex', 'Embarked', 'Survived']
df_ = df[include]  # only using 4 variables

categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
    else:
          df_[col].fillna(0, inplace=True)

df_ohe = pd.get_dummies(df, columns=categoricals, dummy_na=True)

from sklearn.ensemble import RandomForestClassifier as rf
dependent_variable = 'Survived'
x = df_ohe[df_ohe.columns.difference([dependent_variable])
y = df_ohe[dependent_variable]
clf = rf()
clf.fit(x, y)

from sklearn.externals import joblib
joblib.dump(clf, 'model.pkl')
           
clf = joblib.load('model.pkl')

           
from flask import Flask
app = Flask(__name__)
if __name__ == '__main__':
     app.run(port=8080)

 from flask import Flask, jsonify
from sklearn.externals import joblib
import pandas as pd
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_)
     query = pd.get_dummies(query_df)
     prediction = clf.predict(query)
     return jsonify({'prediction': list(prediction)})
if __name__ == '__main__':
     clf = joblib.load('model.pkl')
     app.run(port=8080)          

           
model_columns = list(x.columns)
joblib.dumps(model_columns, 'model_columns.pkl')
 

@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_)
     query = pd.get_dummies(query_df)
     for col in model_columns:
          if col not in query.columns:
               query[col] = 0
     prediction = clf.predict(query)
     return jsonify({'prediction': list(prediction)})
if __name__ == '__main__':
     clf = joblib.load('model.pkl')
     model_columns = joblib.load('model_columns.pkl')
     app.run(port=8080)

           


# In[ ]:




