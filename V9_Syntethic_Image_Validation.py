
# %% Cell
import ee
from osgeo import gdal
from osgeo import osr
import pandas as pd
import numpy as np
import time
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

#from save_gee_image_tiff import save_gee_tiff
ee.Initialize()

# %% Cell
import os
val_path = './simulation2.tif' if os.path.exists('./simulation2.tif') else '../../../idl/Default/simulation2.tif'
validation = rasterio.open(val_path)
show(validation.read(4))

# %% Cell
composite = np.dstack((validation.read(20), validation.read(19), validation.read(18)))
plt.imshow(composite)

# %% Cell
import math
shp = validation.read(10).shape[1]/2
c = int(shp)
area = 400*400
#training_pixels = validation.read()[:,int(c-(np.sqrt(area)/2)):int(c+(np.sqrt(area)/2)),int(c-(np.sqrt(area)/2)):int(c+(np.sqrt(area)/2))]
training_pixels = validation.read()[:,400:600,400:600]
tp = training_pixels.reshape(-1,1)
len(tp)

# %% Cell
beta = float(input('digite o valor de beta: '))
new_dataset = int(beta * len(tp))
dataind = np.random.choice(len(tp), size=new_dataset)
tp_opt = tp[dataind]
len(tp_opt)

# %% Cell
import math
shp = validation.read(10).shape[1]/2
c = int(shp)
area = 400*400
x = np.zeros(validation.read(10).shape)
for i in ([6,5,4,3,2,1]):
    x[int(c-(np.sqrt(i*area)/2)):int(c+(np.sqrt(i*area)/2)+1),int(c-(np.sqrt(i*area)/2)):int(c+(np.sqrt(i*area)/2)+1)] = i-1
    
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
base_df = pd.DataFrame(index=labels)
#base_df

# %% Cell
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def metricas(x,tab_ex2):
    sm = tab_ex2.sum(axis=1).values.reshape(979,979)
    valida = np.where(sm<=0, int(1), sm)
    valida = np.where(((valida>0)&(valida<=20)), int(1), valida)
    valida = np.where(((valida>20)&(valida<=40)), int(2), valida)
    valida = np.where((valida>40)&(valida<=60), int(3), valida)
    valida = np.where(((valida>60)&(valida<=80)), int(4), valida)
    valida = np.where(((valida>80)&(valida<=100)), int(5), valida)
    valida_teste = valida.copy()
    valida_teste[int(c-(np.sqrt(area)/2)):int(c+(np.sqrt(area)/2)+1),int(c-(np.sqrt(area)/2)):int(c+(np.sqrt(area)/2)+1)] = 0
    valida_teste = valida_teste.reshape(-1,1)
    new_valida = np.delete(valida_teste, np.where(valida_teste==0))
    xx = x.reshape(-1,1)
    new_x = np.delete(xx, np.where(xx==0))
    valor = f1_score(new_x.reshape(-1,1),new_valida.reshape(-1,1), average=None)
    return valor

def isola(df, classificador):
    x = df.to_numpy().reshape(-1,1)
    res2 = classificador.predict(x)
    res = np.where(res2 < 0, abs(2*res2), 0)
    ser = pd.Series(res, index=df.index)
    return ser

def cria_dataf(x, classificador):
    tab_ex = pd.DataFrame(index=x.index, columns= x.columns)

    for i in range(x.shape[1]):
        vals = isola(x.iloc[:,i], classificador)
        tab_ex.iloc[:,i] = vals  
    return tab_ex

# %% Cell
pd.options.mode.chained_assignment = None
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

teste = validation.read().reshape([100,-1])[:50]
t = pd.DataFrame(teste.T)
comp_df_oc = base_df.copy()
t.head()

# %% Cell
#comp_df_oc = base_df.copy()
dict_atua = {'OCSVM_nu_0.025': [0.951038, 0.651241, 0.105222, 0.010581, 0.000103],
            'OCSVM_nu_0.05': [0.840595, 0.870718, 0.990308, 0.994402, 0.994279],
            'OCSVM_nu_0.1': [0.342966, 0.616279, 0.902855, 0.991923, 0.994247] }
chaves = list(dict_atua.keys())
for i in range(3):
    ser = pd.Series(dict_atua[chaves[i]], index=base_df.index)
    comp_df_oc.loc[:,chaves[i]] = ser
    
comp_df_oc

# %% Cell
#nuu = [0.001, 0.005, 0.01, 0.05, 0.1]
nuu = [0.25, 0.5] #[0.05,0.1,0.25,0.5] #[0.025]
data_dict = {}
for i in nuu:
    a = time.time()
    classificador = OneClassSVM(nu=i, kernel='rbf', gamma='auto').fit(tp_opt)
    b = time.time()
    print(str(b-a) + ' segundos')
    tab_ex = cria_dataf(t, classificador)
    d = time.time()
    nest = "OCSVM_nu_" + str(i)
    print(nest + " " + str((d-b)/60) + ' minutos')
    ser = pd.Series(metricas(x,tab_ex), index=base_df.index)
    comp_df_oc.loc[:,nest] = ser
    data_dict[i] = tab_ex
    display(comp_df_oc.head())

# %% Cell
comp_df_oc

# %% Cell
comp_df_oc.plot(title='f1 score by Anomaly Probability', xlabel='Anomaly probability', ylabel='f1 score',linestyle='--', marker='o')

# %% Cell
comp_df_oc.to_csv('F1_Score_OCSVM_Data.csv')

# %% Cell
comp_df_oc_v2.plot(title='f1 score by Anomaly Probability', xlabel='Anomaly probability', ylabel='f1 score',linestyle='--', marker='o')

# %% Cell
import matplotlib as mpl
sm = data_dict[0.05].sum(axis=1).values.reshape(979,979)

fig, ax = plt.subplots()

cmap = mpl.cm.turbo
bounds = [0, 20, 40, 60, 80, 100]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, )
pcm = ax.pcolormesh(sm, norm=norm, cmap=cmap)
fig.colorbar(pcm)
plt.title('OCSVM anomaly probability')
plt.show()

# %% Cell
comp_df_oc.plot(title='f1 score by Anomaly Probability', xlabel='Anomaly probability', ylabel='f1 score',linestyle='--', marker='o')

# %% Cell
nuu = [0.001, 0.005, 0.01, 0.05, 0.1]
comp_df_oc2 = base_df.copy()
for i in nuu:
    classificador = OneClassSVM(nu=i).fit(tp_opt)
    print(i)
    tab_ex = cria_dataf(t, classificador)
    print(i)
    nest = "OCSVM_nu_" + str(i)
    ser = pd.Series(metricas(x,tab_ex), index=base_df.index)
    comp_df_oc2.loc[:,nest] = ser
    print('foi')
    
comp_df_oc2

# %% Cell
comp_df_oc2

# %% Cell
import math
shp = validation.read(10).shape[1]/2
c = int(shp)
area = 400*400
x = np.zeros(validation.read(10).shape)
for i in ([6,5,4,3,2,1]):
    x[int(c-(np.sqrt(i*area)/2)):int(c+(np.sqrt(i*area)/2)+1),int(c-(np.sqrt(i*area)/2)):int(c+(np.sqrt(i*area)/2)+1)] = i-1
    
x

# %% Cell
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
base_df = pd.DataFrame(index=labels)
base_df

# %% Cell
pd.options.mode.chained_assignment = None
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

teste = validation.read().reshape([100,-1])
t = pd.DataFrame(teste.T)
inicio = 20
comp_df = base_df.copy()

for i in range(5):
    valor = inicio + i*20
    classificador = IsolationForest(n_estimators=valor,random_state=0,
                                n_jobs=-1,max_samples='auto').fit(tp)
    
    tab_ex = cria_dataf(t, classificador)
    print(i)
    nest = "IF_" + str(valor)
    ser = pd.Series(metricas(x,tab_ex), index=base_df.index)
    comp_df.loc[:,nest] = ser
    print('foi')
    
comp_df

# %% Cell
ocsvm = [0.813267, 0.858453, 0.992979, 0.994383, 0.994279]
ser = pd.Series(ocsvm, index=base_df.index)
comp_df.loc[:, "One-Class SVM"] = ser

# %% Cell
#oioi = base_copy['One-Class SVM'].values
#tchau = pd.Series(oioi, index=base_df.index)
#comp_df.loc[:,'One-Class SVM'] = tchau
comp_df.plot(title='f1 score by Anomaly Probability', xlabel='Anomaly probability', ylabel='f1 score',linestyle='--', marker='o')

# %% Cell

