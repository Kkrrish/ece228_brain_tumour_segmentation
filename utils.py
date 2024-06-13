import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt

def load_dataset(path):
    p = os.listdir(path)
    p.sort(key=str.lower)
    arr = []
    for i in range(len(p)):
        if(i != 4):
            p1 = os.listdir(path+'/'+p[i])
            p1.sort()
            img = sitk.ReadImage(path+'/'+p[i]+'/'+p1[-1])
            arr.append(sitk.GetArrayFromImage(img))
        else:
            p1 = os.listdir(path+'/'+p[i])
            img = sitk.ReadImage(path+'/'+p[i]+'/'+p1[0])
            Y_labels = sitk.GetArrayFromImage(img)

    data = np.zeros((Y_labels.shape[1],Y_labels.shape[0],Y_labels.shape[2],4))
    for i in range(Y_labels.shape[1]):
        data[i,:,:,0] = arr[0][:,i,:]
        data[i,:,:,1] = arr[1][:,i,:]
        data[i,:,:,2] = arr[2][:,i,:]
        data[i,:,:,3] = arr[3][:,i,:]
    return data, Y_labels
