
# %% Cell
import numpy as np
import rasterio
import ee
import fiona
import rasterio.mask
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


ee.Initialize()

# %% Cell
from matplotlib.pyplot import figure

def aplica_labels(df1, maximo):
    df1.loc[df1["anomalies"]<int(0.2*maximo),'label'] = "Very low"
    df1.loc[(df1["anomalies"]<int(0.4*maximo)) & (df1["anomalies"]>=int(0.2*maximo)),'label'] = "Low"
    df1.loc[(df1["anomalies"]<int(0.6*maximo)) & (df1["anomalies"]>=int(0.4*maximo)),'label'] = "Medium"
    df1.loc[(df1["anomalies"]<int(0.8*maximo)) & (df1["anomalies"]>=int(0.6*maximo)),'label'] = "High"
    df1.loc[df1["anomalies"]>=int(0.8*maximo),'label'] = "Very high"
    df1.reset_index(inplace=True, drop=True)
    df1.loc[df1['label']=="Very high", 'id'] = 5
    df1.loc[df1['label']=="High", 'id'] = 4
    df1.loc[df1['label']=="Medium", 'id'] = 3
    df1.loc[df1['label']=="Low", 'id'] = 2
    df1.loc[df1['label']=="Very low", 'id'] = 1
    return df1

def gera_data(link_shp,link_shp2,link_tiff):
    with fiona.open(link_shp) as shape_afetadas:
        afetadas = [feature["geometry"] for feature in shape_afetadas]

    with rasterio.open(link_tiff) as regress:
        afetadas_regress, out_transform = rasterio.mask.mask(regress, afetadas, crop=True)
        out_meta = regress.meta

    out_meta.update({"driver": "GTiff",
                     "height": afetadas_regress.shape[1],
                     "width": afetadas_regress.shape[2],
                     "transform": out_transform})

    #with rasterio.open("afetadas.tif", "w", **out_meta) as dest:
     #   dest.write(afetadas_regress)

    df1 = pd.DataFrame(columns=['anomalies'],data=list(zip(afetadas_regress.flatten())))
    remove_index = df1[df1['anomalies']<1].index
    df1 = df1.drop(remove_index)
    maximo = df1.max().values
    df1 = aplica_labels(df1, maximo)
    
    with fiona.open(link_shp2) as shape_afetadas2:
        afetadas2 = [feature["geometry"] for feature in shape_afetadas2]
        
    with rasterio.open(link_tiff) as regress:
        afetadas_regress2, out_transform2 = rasterio.mask.mask(regress, afetadas2, crop=True)
        out_meta = regress.meta
    
    out_meta.update({"driver": "GTiff",
                     "height": afetadas_regress2.shape[1],
                     "width": afetadas_regress2.shape[2],
                     "transform": out_transform2})
    df2 = pd.DataFrame(columns=['anomalies'],data=list(zip(afetadas_regress2.flatten())))
    remove_index = df2[df2['anomalies']<1].index
    df2 = df2.drop(remove_index)
    df2 = aplica_labels(df2, maximo)
    resu = pd.concat([df1,df2])
    return resu, df1, df2

def histo_gera(df1, df2, year):
    figure(figsize=(14,10), dpi=300)
    titulo = "NDVI Anomalies for Fundão dam, Mariana-MG " + year
    plt.title(titulo, fontsize=25)
    plt.xlabel('Anomaly count', fontsize=25)
    plt.ylabel('Frequency', fontsize=25)
    labels1 = df1.sort_values(by=['id'])['label'].unique()
    t1 = []
    t2 = []
    for label in labels1:
        t1.append(df1.sort_values(by=['id'])['label'].value_counts()[label])

    labels2 = df2.sort_values(by=['id'])['label'].unique()
    for label in labels2:
        t2.append(df2.sort_values(by=['id'])['label'].value_counts()[label])

    yy = t1/np.sum(t1)
    y2 = t2/np.sum(t2)
    xx = np.arange(len(labels1))
    x2 = np.arange(len(labels2))
    width = 0.35
    ax = plt.subplot(111)
    ax.bar(x2 + width/2, y2, width, color='#07051b', label='Non Changes')
    ax.bar(xx - width/2, yy, width, color='#e25734', label='Changes')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks(xx)
    ax.set_xticklabels(labels1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=15)
    ax.legend()
    plt.legend(fontsize=20)
    path = "../../Mariana/PNG/V3_Sep_22/" + titulo + ".png"
    plt.savefig(path)
    return 'ok' + titulo

# %% Cell
def histo_gera2(df1, year):
    figure(figsize=(14,10), dpi=300)
    titulo = "GVMI Anomalies for Fundão dam, Mariana-MG " + year
    plt.title(titulo, fontsize=25)
    plt.xlabel('Anomaly count', fontsize=25)
    plt.ylabel('Frequency', fontsize=25)
    labels1 = df1.sort_values(by=['id'])['label'].unique()
    t1 = []
    for label in labels1:
        t1.append(df1.sort_values(by=['id'])['label'].value_counts()[label])

    yy = t1/np.sum(t1)
    xx = np.arange(len(labels1))
    width = 0.35
    ax = plt.subplot(111)
    ax.bar(xx - width/2, yy, width, color='#e25734', label='Isolation Forest')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks(xx)
    ax.set_xticklabels(labels1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=15)
    ax.legend()
    plt.legend(fontsize=20)
    path = "../../Mariana/PNG/V3_Sep_22/" + titulo + ".png"
    plt.savefig(path)
    return 'ok' + titulo

# %% Cell
def histo_gera3(df1, df2, year):
    figure(figsize=(14,10), dpi=300)
    titulo = "NDVI Anomalies for Fundão dam, Mariana-MG " + year
    plt.title(titulo, fontsize=25)
    plt.xlabel('Anomaly count', fontsize=25)
    plt.ylabel('Frequency', fontsize=25)
    labels1 = df1.sort_values(by=['id'])['label'].unique()
    t1 = []
    t2 = []
    for label in labels1:
        t1.append(df1.sort_values(by=['id'])['label'].value_counts()[label])

    labels2 = df2.sort_values(by=['id'])['label'].unique()
    for label in labels2:
        t2.append(df2.sort_values(by=['id'])['label'].value_counts()[label])

    yy = t1/np.sum(t1)
    y2 = t2/np.sum(t2)
    xx = np.arange(len(labels1))
    x2 = np.arange(len(labels2))
    width = 0.35
    ax = plt.subplot(111)
    ax.bar(x2 + width/2, y2, width, color='#07051b', label='Non Changes')
    ax.bar(xx - width/2, yy, width, color='#e25734', label='Changes')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks(xx)
    ax.set_xticklabels(labels1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=15)
    ax.legend()
    plt.legend(fontsize=20)
    path = "../../Mariana/PNG/V3_Sep_22/" + titulo + ".png"
    plt.savefig(path)
    return 

# %% Cell
import os
bru_shp_path = "./qgis/Barragens/Mariana/v2/RatroLamaMari.shp" if os.path.exists("./qgis/Barragens/Mariana/v2/RatroLamaMari.shp") else "../../../qgis/Barragens/Mariana/v2/RatroLamaMari.shp"
path = "../../Mariana/Tiff/V4_Out_22/"
inter_anos = []
dict_if_path = {}
for x in os.listdir(path):
    if x.endswith(".tif"):
        ano = x[13:-4] + "_Estimators"
        inter_anos.append(ano)
        dict_if_path[ano] = path + x
        
dict_if_path

# %% Cell
dict_ocsvm_path = {}
nuu = [0.001, 0.005, 0.01, 0.05, 0.1]
cont = 0
path = "../../Mariana/Tiff/V3_Sep_22/OCSVM/"
for x in os.listdir(path):
    if x.endswith(".tif"):
        ano = str(nuu[cont]) + "_nu"
        inter_anos.append(ano)
        dict_ocsvm_path[ano] = path + x
        cont += 1
        
dict_ocsvm_path

# %% Cell
bru_shp_path_no = "./qgis/Barragens/Mariana/v2/unchangedareas_mari.shp" if os.path.exists("./qgis/Barragens/Mariana/v2/unchangedareas_mari.shp") else "../../../qgis/Barragens/Mariana/v2/unchangedareas_mari.shp"
dicionario_labels = {}
dt_dict_oc = {}

for t in dict_ocsvm_path:
    dfx, df1, df2 = gera_data(bru_shp_path,bru_shp_path_no,dict_ocsvm_path[t])
    dfoc_0 = aplica_labels(df1,64)
    nome1 = 'changes_' + str(t)
    dicionario_labels[nome1] = dfoc_0
    dfoc_1 = aplica_labels(df2,64)
    nome2 = 'unchanges_' + str(t) 
    dicionario_labels[nome2] = dfoc_1
    dt_dict_oc[t] = dfx
    histo_gera(dfoc_0,dfoc_1,t)
    plt.show()

# %% Cell
dicionario_labels.keys()

# %% Cell
titles = list(dicionario_labels.keys())
uy2 = pd.DataFrame()
for i in titles:
    okl = dicionario_labels[i]['label'].value_counts()
    t = okl.sum()
    okl = okl/t
    uy = pd.DataFrame(okl)
    uy = uy.fillna(0)
    uy = uy.set_axis([i], axis=1, inplace=False)
    uy2 = pd.concat([uy,uy2], axis=1)

# %% Cell
saida = uy2.fillna(0)
saida.to_csv('../../Mariana/csv/hist_bars_OCSVM_Mariana.csv')

# %% Cell
dicio_lab_if = {}

for t in dict_if_path:
    dfy, dfy1, dfy2 = gera_data(bru_shp_path,bru_shp_path_no, dict_if_path[t])
    dfif_0 = aplica_labels(dfy1,64)
    nome1 = 'changes_' + str(t)
    dicio_lab_if[nome1] = dfif_0
    dfif_1 = aplica_labels(dfy2,64)
    nome2 = 'unchanges_' + str(t) 
    dicio_lab_if[nome2] = dfif_1
    histo_gera(dfif_0,dfif_1,t)
    plt.show()

# %% Cell
titles = list(dicio_lab_if.keys())
uy2 = pd.DataFrame()
for i in titles:
    okl = dicio_lab_if[i]['label'].value_counts()
    t = okl.sum()
    okl = okl/t
    uy = pd.DataFrame(okl)
    uy = uy.fillna(0)
    uy = uy.set_axis([i], axis=1, inplace=False)
    uy2 = pd.concat([uy,uy2], axis=1)

saida = uy2.fillna(0)
saida

# %% Cell
saida.to_csv('../../Mariana/csv/hist_bars_IF_Mariana.csv')

# %% Cell
teste = list(dicionario_labels.keys())
teste

# %% Cell
def inicio_matcon(df1):
    df1.loc[df1['label']=="Very high", 'name'] = 'Changed'
    df1.loc[df1['label']=="High", 'name'] = 'Changed'
    df1.loc[df1['label']=="Medium", 'name'] = 'Transition'
    df1.loc[df1['label']=="Low", 'name'] = 'Unchanged'
    df1.loc[df1['label']=="Very low", 'name'] = 'Unchanged'
    return df1

# %% Cell
def accuracy(df):
    soma = 0
    for i in range(len(df)):
        soma += df.iloc[i,i]
        
    res = soma/df.sum().sum()
    return res

def norm_accuracy(df):
    soma = 0
    for i in range(len(df)):
        soma += df.iloc[i,i]
        
    res = soma/2
    return res

def random_accuracy(df):
    soma = 0
    for i in range(len(df)):
        soma += df.iloc[0,i]
        
    p1 = accuracy(df)
    p2 = soma/df.sum().sum()
    res = p1*p2 + (1-p1)*(1-p2)
    return res

def norm_random_accuracy(df):
    soma = 0
    for i in range(len(df)):
        soma += df.iloc[0,i]
        
    p1 = norm_accuracy(df)
    p2 = soma/2
    res = p1*p2 + (1-p1)*(1-p2)
    return res

def kappa(df):
    res = (accuracy(df) - random_accuracy(df))/(1-random_accuracy(df))
    return res

def norm_kappa(df):
    res = (norm_accuracy(df) - norm_random_accuracy(df))/(1 - norm_random_accuracy(df))
    return res

# %% Cell
dict_matcon = {}
data_metrica = pd.DataFrame()
for i in range(int(len(teste)/2)):
    df1 = dicionario_labels[teste[2*i]]
    df1 = inicio_matcon(df1)
    df2 = dicionario_labels[teste[2*i+1]]
    df2 = inicio_matcon(df2)
    ni = teste[2*i][8:]
    cfm_if_chgs = df1.drop(df1[df1['name']=='Transition'].index)
    cfm_if_nchgs = df2.drop(df2[df2['name']=='Transition'].index)
    vl_l = cfm_if_nchgs['name'].value_counts()/len(df2)
    vh_h = cfm_if_chgs['name'].value_counts()/len(df1)
    cm_df_if = pd.DataFrame([vh_h,vl_l], index = ['Sample of changes', 'Sample of no changes']).fillna(0)
    dict_matcon[i] = cm_df_if
    data_metrica.loc[ni, "Accuracy"] = accuracy(cm_df_if)
    data_metrica.loc[ni, "Kappa"] = kappa(cm_df_if)

# %% Cell
data_metrica

# %% Cell
data_metrica.to_csv('../../Mariana/csv/kappa_ocsvm_mariana.csv')

# %% Cell
#Plotting the confusion matrix
import seaborn as sns
for i in dict_matcon:
    plt.figure(figsize=(7,5))
    sns.heatmap(dict_matcon[i], annot=True)
    plt.title('Mariana Confusion Matrix OCSVM ' + str(i))
    plt.show()

# %% Cell
teste = list(dicio_lab_if.keys())
teste

# %% Cell
cm_df_if

# %% Cell
dict_matcon_if = {}
data_metrica_if = pd.DataFrame()
teste = list(dicio_lab_if.keys())

for i in range(int(len(teste)/2)):
    df1 = dicio_lab_if[teste[2*i]]
    df1 = inicio_matcon(df1)
    df2 = dicio_lab_if[teste[2*i+1]]
    df2 = inicio_matcon(df2)
    ni = teste[2*i][8:]
    cfm_if_chgs = df1.drop(df1[df1['name']=='Transition'].index)
    cfm_if_nchgs = df2.drop(df2[df2['name']=='Transition'].index)
    vl_l = cfm_if_nchgs['name'].value_counts()/len(df2)
    vh_h = cfm_if_chgs['name'].value_counts()/len(df1)
    cm_df_if = pd.DataFrame([vh_h,vl_l], index = ['Sample of changes', 'Sample of no changes']).fillna(0)
    dict_matcon_if[i] = cm_df_if
    data_metrica_if.loc[ni, "Accuracy"] = accuracy(cm_df_if)
    data_metrica_if.loc[ni, "Kappa"] = kappa(cm_df_if)

# %% Cell
data_metrica_if

# %% Cell
data_metrica_if.to_csv('../../Mariana/csv/kappa_if_mariana.csv')

# %% Cell
import seaborn as sns
for i in dict_matcon_if:
    plt.figure(figsize=(7,5))
    sns.heatmap(dict_matcon_if[i], annot=True)
    plt.title('Mariana Confusion Matrix IF ' + str(i))
    plt.show()

# %% Cell
data_metrica['Kappa'].plot(title='kappa score by contamination Mariana', xlabel='fraction of dataset', ylabel='kappa',linestyle='--', marker='o')

# %% Cell
data_metrica_if['Kappa'].plot(title='kappa score by estimators Mariana', xlabel='fraction of dataset', ylabel='kappa',linestyle='--', marker='o',color='red')

# %% Cell
with fiona.open("ic.gino/qgis/Barragens/Brumadinho/changes.shp") as shape_afetadas:
    afetadas = [feature["geometry"] for feature in shape_afetadas]

# %% Cell
with rasterio.open("ic.gino/python/Brumadinho/Tiff/Bru_NDWI_OC_sentinel.tif") as regress:
    afetadas_regress, out_transform = rasterio.mask.mask(regress, afetadas, crop=True)
    out_meta = regress.meta

# %% Cell
out_meta.update({"driver": "GTiff",
                 "height": afetadas_regress.shape[1],
                 "width": afetadas_regress.shape[2],
                 "transform": out_transform})

with rasterio.open("afetadas.tif", "w", **out_meta) as dest:
    dest.write(afetadas_regress)

# %% Cell
afetadas_regress.flatten()

# %% Cell
df1 = pd.DataFrame(columns=['anomalies'],data=list(zip(afetadas_regress.flatten())))

# %% Cell
remove_index = df1[df1['anomalies']<1].index

# %% Cell
df1 = df1.drop(remove_index)

# %% Cell
df1.head()

# %% Cell
maximo = df1.max().values
maximo

# %% Cell
df1.loc[df1["anomalies"]<int(0.2*maximo),'label'] = "Very low"
df1.loc[(df1["anomalies"]<int(0.4*maximo)) & (df1["anomalies"]>=int(0.2*maximo)),'label'] = "Low"
df1.loc[(df1["anomalies"]<int(0.6*maximo)) & (df1["anomalies"]>=int(0.4*maximo)),'label'] = "Medium"
df1.loc[(df1["anomalies"]<int(0.8*maximo)) & (df1["anomalies"]>=int(0.6*maximo)),'label'] = "High"
df1.loc[df1["anomalies"]>=int(0.8*maximo),'label'] = "Very high"

# %% Cell
df1.reset_index(inplace=True, drop=True)

# %% Cell
df1.loc[df1['label']=="Very high", 'id'] = 5
df1.loc[df1['label']=="High", 'id'] = 4
df1.loc[df1['label']=="Medium", 'id'] = 3
df1.loc[df1['label']=="Low", 'id'] = 2
df1.loc[df1['label']=="Very low", 'id'] = 1
df1.sort_values(by=['id'])

# %% Cell
with fiona.open("ic.gino/qgis/Barragens/Brumadinho/nonchanges.shp") as shape_nafetadas:
    nafetadas = [feature["geometry"] for feature in shape_nafetadas]

# %% Cell
with rasterio.open("ic.gino/python/Brumadinho/Tiff/Bru_NDWI_OC_sentinel.tif") as regress:
    nafetadas_regress, nout_transform = rasterio.mask.mask(regress, nafetadas, crop=True)
    nout_meta = regress.meta

# %% Cell
nout_meta.update({"driver": "GTiff",
                 "height": nafetadas_regress.shape[1],
                 "width": nafetadas_regress.shape[2],
                 "transform": nout_transform})

with rasterio.open("nafetadas.tif", "w", **out_meta) as dest:
    dest.write(nafetadas_regress)

# %% Cell
nafetadas_regress.flatten()

# %% Cell
df2 = pd.DataFrame(columns=['anomalies'],data=list(zip(nafetadas_regress.flatten())))

# %% Cell
remove_index = df2[df2['anomalies']<1].index
df2 = df2.drop(remove_index)
df2.loc[df2["anomalies"]<int(0.2*maximo),'label'] = "Very low"
df2.loc[(df2["anomalies"]<int(0.4*maximo)) & (df2["anomalies"]>=int(0.2*maximo)),'label'] = "Low"
df2.loc[(df2["anomalies"]<int(0.6*maximo)) & (df2["anomalies"]>=int(0.4*maximo)),'label'] = "Medium"
df2.loc[(df2["anomalies"]<int(0.8*maximo)) & (df2["anomalies"]>=int(0.6*maximo)),'label'] = "High"
df2.loc[df2["anomalies"]>=int(0.8*maximo),'label'] = "Very high"
df2.loc[df2['label']=="Very high", 'id'] = 5
df2.loc[df2['label']=="High", 'id'] = 4
df2.loc[df2['label']=="Medium", 'id'] = 3
df2.loc[df2['label']=="Low", 'id'] = 2
df2.loc[df2['label']=="Very low", 'id'] = 1
df2.sort_values(by=['id'])

# %% Cell
from matplotlib.pyplot import figure
figure(figsize=(14,10), dpi=300)

plt.title('One-Class SVM NDWI Anomalies for Feijão Dam, Brumadinho-MG', fontsize=25)
plt.xlabel('Anomaly count', fontsize=25)
plt.ylabel('Frequency', fontsize=25)

labels1 = df1.sort_values(by=['id'])['label'].unique()
t1 = []
t2 = []
for label in labels1:
    t1.append(df1.sort_values(by=['id'])['label'].value_counts()[label])
    
correct_labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']
aux = [0,0,0,0,0]
cont = 0
for i in range(len(correct_labels)):
    if (correct_labels[i] in labels1)==True:
        aux[i] = t1[cont]
        cont = cont + 1
    
labels2 = df2.sort_values(by=['id'])['label'].unique()
for label in labels2:
    t2.append(df2.sort_values(by=['id'])['label'].value_counts()[label])
    
aux2 = [0,0,0,0,0]
cont = 0
for i in range(len(correct_labels)):
    if (correct_labels[i] in labels2)==True:
        aux2[i] = t2[cont]
        cont = cont + 1

yy = aux/np.sum(aux)
y2 = aux2/np.sum(aux2)
xx = np.arange(len(correct_labels))
x2 = np.arange(len(correct_labels))
width = 0.35

ax = plt.subplot(111)
ax.bar(x2 + width/2, y2, width, color='b', label='Unchanged areas')
ax.bar(xx - width/2, yy, width, color='r', label='Changed areas')


plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
ax.set_xticks(xx)
ax.set_xticklabels(correct_labels)
plt.xticks(fontsize=20)
plt.yticks(fontsize=15)
ax.legend()
plt.legend(fontsize=20)
plt.savefig('ic.gino/python/Brumadinho/graphs/OCSVM_NDWI_Bru.svg')

# %% Cell
df1.loc[df1['label']=="Very high", 'name'] = 'Changed'
df1.loc[df1['label']=="High", 'name'] = 'Changed'
df1.loc[df1['label']=="Medium", 'name'] = 'Transition'
df1.loc[df1['label']=="Low", 'name'] = 'Unchanged'
df1.loc[df1['label']=="Very low", 'name'] = 'Unchanged'

try:
    df2.loc[df2['label']=="Very high", 'name'] = 'Changed'
    df2.loc[df2['label']=="High", 'name'] = 'Changed'
    df2.loc[df2['label']=="Medium", 'name'] = 'Transition'
    df2.loc[df2['label']=="Low", 'name'] = 'Unchanged'
    df2.loc[df2['label']=="Very low", 'name'] = 'Unchanged'
except:
    print('não deu')

cfm_ocsvm_chgs = df1.drop(df1[df1['name']=='Transition'].index)
cfm_ocsvm_nchgs = df2.drop(df2[df2['name']=='Transition'].index)

vl_l_o = cfm_ocsvm_nchgs['name'].value_counts()/len(df2)
vh_h_o = cfm_ocsvm_chgs['name'].value_counts()/len(df1)

# %% Cell
display(vl_l,vh_h)

# %% Cell
cm_df_oc = pd.DataFrame([vh_h_o,vl_l_o], index = ['Sample of changes', 'Sample of no changes']).fillna(0)

#Plotting the confusion matrix
import seaborn as sns
plt.figure(figsize=(10,8))
sns.heatmap(cm_df_oc, annot=True)
plt.title('Brumadinho Confusion Matrix One_Class SVM')
plt.show()

# %% Cell
print("Accuracy", accuracy(cm_df_oc), norm_accuracy(cm_df_oc))
print("Kappa", kappa(cm_df_oc), norm_kappa(cm_df_oc))
