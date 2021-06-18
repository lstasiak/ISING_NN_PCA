import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical


import numpy as np
import pandas as pd
import glob

def normalize(X, inverse=False):
    """
    Normalize data using sklearn: MinMaxScaler
    If inverse == True --> Data are re-normalized
    """
    scaler = MinMaxScaler()
    scaler.fit(X)
    
    if inverse:
        return scaler.inverse_transform(X)
    
    return scaler.transform(X)


def base_prepare(df, exportLabels = False, normalize_data = False, select_val_set=True, nsamples=100, index_="L_", save=False):
    if type(df)==str:
        df = pd.read_csv(df, sep=" ", header=None)
    df.dropna(axis=1, inplace=True)
    df.rename(columns={0: "Temperature"}, inplace=True)
    X = df.iloc[:, 1:].values
    
    if exportLabels:
        temp_class = [1 if i>2.26 else 0 for i in df.Temperature]
        Y = to_categorical(temp_class).astype(np.int8)
        np.save(f"labels/labels_{index_}", Y)
    else:
        try:
            Y = np.load(f"labels/labels_{index_}.npy").astype(np.int8)
        except FileNotFoundError: 
            print(f"File 'labels/labels_{index_}.npy' not found!")
            return 0
    
    if normalize_data:
        X = normalize(X)
    
    if save:
        X.to_csv(f"prepared_X_{index_}.csv")
        pd.DataFrame(Y).to_csv(f"prepared_Y_{index_}.csv")
        
    if select_val_set:
        validation_df = select_test_dataset(df, nsamples=nsamples)
        del df
        return (X, Y, validation_df)
    
    del df
    return X, Y


def fit_linregress(x, y, return_stats=False, expand=True, x_factor=0.9, y_factor=1.1):
    """
    fit linear regression using scipy stats
    """

    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress([x,y])
    
    y_fit = slope * x + intercept
    
    print("slope = ", slope)
    print("intercept = ", intercept)
    print("R = ", r_value)
    print("p = ", p_value)
    print("Standard error = ", std_err)
    if not expand:
        if not return_stats:
            return (x, y_fit)
        else:
            return(x, y_fit, [slope, intercept, r_value, p_value, std_err])
    else:
        polynomial = np.array([slope, intercept])
        x_low = x_factor*min(x)
        x_high = y_factor*max(x)
        x_extended = np.linspace(x_low, x_high, 100)
        y_fit = np.polyval(polynomial,x_extended)
        if not return_stats:
            return (x_extended, y_fit)
        else:
            return(x_extended, y_fit, [slope, intercept, r_value, p_value, std_err])


def select_test_dataset(df, nsamples=100):
    """
    select nsamples randomly from given dataframe
    for every distinct Temperature and return as new dataframe
    """
    distinct_temperatures = df.Temperature.unique()
    test_set = pd.DataFrame()

    for t in distinct_temperatures:
        test_set = pd.concat([test_set, df[df.Temperature==t].sample(n=nsamples)], ignore_index=True)
    
    del distinct_temperatures
    return test_set


def collect_all_predictions(prediction_data_paths: list, system_sizes: list, filename=""):
    all_data = pd.DataFrame(columns=["Temperature", "P_low", "P_high", "std_low", "std_high", "L"])
    
    for path, L in zip(prediction_data_paths, system_sizes):
        data = pd.read_csv(path, index_col=0)
        data_means = data.groupby("Temperature").mean()
        data_means["L"] = L
        data_means["Temperature"] = data_means.index
        data_means["std_low"] = data.groupby("Temperature").std()["P_low"]
        data_means["std_high"] = data.groupby("Temperature").std()["P_high"]
        all_data = pd.concat([all_data, data_means], ignore_index=True)
        
    all_data.L = all_data.L.astype(int)
    
    if filename:
        all_data.to_csv(filename)
        
    del data, data_means
    return all_data

def interpolated_intercept(x, y1, y2):
    """Find the intercept of two curves, given by the same x data"""

    def intercept(point1, point2, point3, point4):
        """find the intersection between two lines
        the first line is defined by the line between point1 and point2
        the first line is defined by the line between point3 and point4
        each point is an (x,y) tuple.

        So, for example, you can find the intersection between
        intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

        Returns: the intercept, in (x,y) format
        """    

        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x,y

        L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
        L2 = line([point3[0],point3[1]], [point4[0],point4[1]])

        R = intersection(L1, L2)

        return R

    idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
    xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
    return xc,yc   
   

def determine_crossing_points(dataframe, append_zero_value=True, system_sizes=[10, 20, 30, 40, 60]):
    df = dataframe.copy()
    
    interceptions = pd.DataFrame()
    if append_zero_value:
        list_T = [2/np.log(1+np.sqrt(2))]
        list_L = [0] + [1/i for i in system_sizes]
    else:
        list_T = []
        list_L = [1/i for i in system_sizes]

    x = df[df.L==60].Temperature.values

    for l in system_sizes:
        f = df[df.L==l].P_low.values
        g = df[df.L==l].P_high.values
        intercept_x=interpolated_intercept(x, f, g)[0].flatten()
        print(f"intercepts for L={l}: {intercept_x}")
        if len(intercept_x)>1:
            interception = np.mean(intercept_x)
        else:
            interception = intercept_x[0]
        list_T.append(interception)

    interceptions["inv_L"] = list_L
    interceptions["Tc"] = list_T
    
    del df, f, g
    return interceptions.sort_values(by="inv_L")


def generate_predictions(test_df, model, filename, batch_size=400, only_mean=True):
    """
    input: test set - dataFrame with Temperature column and LxL columns for configurations
    (one row consists of the Temperature and single spin configuration LxL.
           model - neural network keras model
           filename - str
           only_mean - Boolean - if True, return and saves dataframe with data 
           grouped by temperature and aggregated by mean
    output: new dataframe with the predictions made by model on test_df
    """

    predictions_df = pd.DataFrame(columns=["Temperature", "P_low", "P_high"])
    predictions_df["Temperature"] = test_df["Temperature"]

    predictions = model.predict(
        test_df.iloc[:,1:].values, batch_size=batch_size, verbose=1
    )
    predictions_df["P_low"] = predictions[:,0]
    predictions_df["P_high"] = predictions[:,1]
    
    if only_mean:
        predictions_df = predictions_df.groupby("Temperature").mean()
    
    predictions_df.to_csv(filename)
    
    return predictions_df


# def generate_predictions(test_df, test_df_pca, model, filename, batch_size=400, only_mean=True):
#     """
#     input: test set - dataFrame with Temperature column and LxL columns for configurations
#     (one row consists of the Temperature and single spin configuration LxL.
#            model - neural network keras model
#            filename - str
#            only_mean - Boolean - if True, return and saves dataframe with data 
#            grouped by temperature and aggregated by mean
#     output: new dataframe with the predictions made by model on test_df
#     """

#     predictions_df = pd.DataFrame(columns=["Temperature", "P_low", "P_high"])
#     predictions_df["Temperature"] = test_df["Temperature"]

#     predictions = model.predict(
#         test_df_pca, batch_size=batch_size, verbose=1
#     )
#     predictions_df["P_low"] = predictions[:,0]
#     predictions_df["P_high"] = predictions[:,1]
    
#     if only_mean:
#         predictions_df = predictions_df.groupby("Temperature").mean()
    
#     predictions_df.to_csv(filename)
    
#     return predictions_df
    

def dense_block(x, hidden_units, activation, l2_reg, bias_factor, dropRate=0):
    output = Dense(hidden_units, activation=activation,
                   kernel_regularizer=l2(l2_reg), bias_regularizer=l2(bias_factor))(x)
    if dropRate:
        output = Dropout(dropRate)(output)
    
    return output
    
def build_func_model(input_shape, 
                     hidden_units=100, 
                     activation="relu", 
                     num_classes=2, 
                     l2_reg=0.1,
                     bias_factor=0.001):
    inputs = keras.Input(shape=input_shape)
    x = dense_block(inputs, hidden_units, activation, l2_reg, bias_factor, dropRate=0)
    x = dense_block(x, hidden_units, activation, l2_reg, bias_factor, dropRate=0)

    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def build_sequential_model(input_shape=(100,), hidden_units=100, hidden_layers=1,
               activation="relu", reg_factor=0.001, bias_factor=0.001, num_classes=2, compile=True):
    print('C R E A T E   M O D E L')
    model = Sequential()
    model.add(Dense(hidden_units, activation=activation, input_shape=input_shape)) # input layer
    
    for i in range(hidden_layers):
        model.add(Dense(hidden_units, activation=activation, kernel_regularizer=l2(reg_factor), bias_regularizer=l2(bias_factor)))
    
    model.add(Dense(num_classes, activation='softmax'))    
    
    if compile:
        model.compile(loss='categorical_crossentropy',optimizer="adam", metrics='acc')
    
    return model