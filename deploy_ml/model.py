import tensorflow as tf
import pandas as pd
import pickle


def predict_model(crim, zn, indus, chas,
                  nox, rm, age, dis, rad,
                  tax, ptratio, black, lstat):

    # Import variable
    scaler_x = pickle.load(open('./saved_model/scaler_x.pickle', 'rb'))
    scaler_y = pickle.load(open('./saved_model/scaler_y.pickle', 'rb'))
    model = tf.keras.models.load_model('./saved_model/model.h5')

    # Make data
    data = {
        'CRIM': [crim],
        'ZN': [zn],
        'INDUS': [indus],
        'CHAS': [chas],
        'NOX': [nox],
        'RM': [rm],
        'AGE': [age],
        'DIS': [dis],
        'RAD': [rad],
        'TAX': [tax],
        'PTRATIO': [ptratio],
        'B': [black],
        'LSTAT': [lstat]
    }

    predict_df = pd.DataFrame(data=data)
    predict_x = scaler_x.transform(predict_df)
    predict_x = model.predict(predict_x)
    predict = scaler_y.inverse_transform(predict_x)
    return predict[0][0]
