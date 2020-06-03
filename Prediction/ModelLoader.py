from sklearn.externals import joblib
import pandas as pd
import numpy as np
import pickle
    
    
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,12)
    loaded_model = pickle.load(open("NNmodel/model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


example_list=[44,3,15,10,1,6,0,2,1,7688,0,40]
print(ValuePredictor(example_list))