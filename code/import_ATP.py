from sklearn.utils import shuffle
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import gc

datalabel="NADP"
MAXSEQ=15
NUMDEPENDENT=7

def MAX_SEQ():
    return MAXSEQ

def NUM_DEPENDENT():
    return NUMDEPENDENT

def data_label():
    return datalabel

def MCNN_data_load(NUM_CLASSES,NUMDEPENDENT):
    path_x_train = "/mnt/D/jupyter/ATPBinding/Dataset/Series" + str(MAXSEQ) + "/Atp388-ABC-CFTR/ATP388-ABC/ProtTrans/data.npy" ## 
    path_y_train = "/mnt/D/jupyter/ATPBinding/Dataset/Series" + str(MAXSEQ) + "/Atp388-ABC-CFTR/ATP388-ABC/ProtTrans/label.npy" ## 
    print(path_x_train)
    print(path_y_train)
    x,y=data_load(path_x_train,path_y_train,NUM_CLASSES)
    path_x_test = "/mnt/D/jupyter/ATPBinding/Dataset/Series" + str(MAXSEQ) + "/Atp388-ABC-CFTR/CFTR/ProtTrans/data.npy"  ##
    path_y_test = "/mnt/D/jupyter/ATPBinding/Dataset/Series" + str(MAXSEQ) + "/Atp388-ABC-CFTR/CFTR/ProtTrans/label.npy" ## 
    print(path_x_test)
    print(path_y_test)
    x_test,y_test=data_load(path_x_test,path_y_test,NUM_CLASSES)
    
    return(x,y,x_test,y_test)

def data_load(x_folder, y_folder,NUM_CLASSES,):
    x_train=np.load(x_folder)
    y_train=np.load(y_folder)
    

    y_train = tf.keras.utils.to_categorical(y_train,NUM_CLASSES)
    gc.collect()
    
    return x_train, y_train