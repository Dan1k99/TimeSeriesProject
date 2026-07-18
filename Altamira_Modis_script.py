
# %% Cell
import ee
from osgeo import gdal
from osgeo import osr
import pandas as pd
import numpy as np
import time

#from save_gee_image_tiff import save_gee_tiff
ee.Initialize()

# %% Cell
def create_reduce_region_function(geometry,
                                  reducer=ee.Reducer.mean(),
                                  scale=30,
                                  crs='EPSG:4326',
                                  bestEffort=True,
                                  maxPixels=1e13,
                                  tileScale=4):

  def reduce_region_function(img):
    stat = img.reduceRegion(
        reducer=reducer,
        geometry=geometry,
        scale=scale,
        crs=crs,
        bestEffort=bestEffort,
        maxPixels=maxPixels,
        tileScale=tileScale)

    return ee.Feature(geometry, stat).set({'millis': img.date().millis()})
  return reduce_region_function

def fc_to_dict(fc):
  prop_names = fc.first().propertyNames()
  prop_lists = fc.reduceColumns(
      reducer=ee.Reducer.toList().repeat(prop_names.size()),
      selectors=prop_names).get('list')

  return ee.Dictionary.fromLists(prop_names, prop_lists)

def add_date_info(df):
  df['Timestamp'] = pd.to_datetime(df['millis'], unit='ms')
  df['Year'] = pd.DatetimeIndex(df['Timestamp']).year
  df['Month'] = pd.DatetimeIndex(df['Timestamp']).month
  df['Day'] = pd.DatetimeIndex(df['Timestamp']).day
  df['DOY'] = pd.DatetimeIndex(df['Timestamp']).dayofyear
  return df

# %% Cell
#altamira_sul = '-55.39299190419731,-8.72942896034137,-54.71183956044731,-6.999099965455131'
altamira_sul = '-55.3809, -7.5173, -54.714, -8.005'
datas = "2010-01-01,2021-12-31"
inicio,fim = datas.split(",")
x1,y1,x2,y2 = altamira_sul.split(",")
geometria = geometry = ee.Geometry.Polygon(
        [[[float(x1),float(y2)],
          [float(x2),float(y2)],
          [float(x2),float(y1)],
          [float(x1),float(y1)],
          [float(x1),float(y2)]]])

sul_Modis = ee.ImageCollection('MODIS/006/MOD13Q1').filterBounds(geometria).filterDate(inicio,fim).select('NDVI','SummaryQA')
print("Modis: "+str(sul_Modis.size().getInfo()))
def ndvi(imagem):
    ndvi = imagem.expression('(NDVI)/10000',{'NDVI':imagem.select(NDVI)}).rename('ndvi')
    return imagem.addBands(ndvi)
NDVI = 'NDVI'
sul_Modis = sul_Modis.map(ndvi)
listaCN = sul_Modis.toList(sul_Modis.size())

# %% Cell
reduce_ndvi = create_reduce_region_function(
    geometry=geometria, reducer=ee.Reducer.mean(), scale=250)

ndvi_stat_fc = ee.FeatureCollection(sul_Modis.map(reduce_ndvi)).filter(
    ee.Filter.notNull(sul_Modis.first().bandNames()))

ndvi_dict = fc_to_dict(ndvi_stat_fc).getInfo()
ndvi_df = pd.DataFrame(ndvi_dict)
display(ndvi_df)
print(ndvi_df.dtypes)

# %% Cell
ndvi_df2 = ndvi_df[ndvi_df['SummaryQA']<1]

# %% Cell
ndvi_df2 = add_date_info(ndvi_df2)
ndvi_df2.tail(5)

# %% Cell
import altair as alt
highlight = alt.selection(
    type='single', on='mouseover', fields=['Year'], nearest=True)

base = alt.Chart(ndvi_df2).encode(
    x=alt.X('DOY:Q', scale=alt.Scale(domain=[0, 353], clamp=True)),
    y=alt.Y('ndvi:Q', scale=alt.Scale(domain=[0, 1])),
    color=alt.Color('Year:O', scale=alt.Scale(scheme='magma')))

points = base.mark_circle().encode(
    opacity=alt.value(0),
    tooltip=[
        alt.Tooltip('Year:O', title='Year'),
        alt.Tooltip('DOY:Q', title='DOY'),
        alt.Tooltip('ndvi:Q', title='NDVI')
    ]).add_selection(highlight)

lines = base.mark_line().encode(
    size=alt.condition(~highlight, alt.value(1), alt.value(3)))

(points + lines).properties(width=600, height=350).interactive()

# %% Cell
base = alt.Chart(ndvi_df2).encode(
    x=alt.X('DOY:Q', scale=alt.Scale(domain=(0, 360))))

line = base.mark_line().encode(
    y=alt.Y('median(ndvi):Q', scale=alt.Scale(domain=(0, 1))))

band = base.mark_errorband(extent='iqr').encode(
    y='ndvi:Q')

(line + band).properties(width=600, height=300).interactive()

# %% Cell
def ext_lat_lon_pixel250(image, geometria, bandas):
    image = image.addBands(ee.Image.pixelLonLat())
    coordenadas = image.select(['longitude', 'latitude']+bandas).reduceRegion(reducer=ee.Reducer.toList(),geometry=geometria,scale=250,bestEffort=True)
    bandas_valores = []
    for banda in bandas:
        bandas_valores.append(np.array(ee.List(coordenadas.get(banda)).getInfo()).astype(float))

    return np.array(ee.List(coordenadas.get('latitude')).getInfo()).astype(float), np.array(ee.List(coordenadas.get('longitude')).getInfo()).astype(float), bandas_valores

# %% Cell
indices = ndvi_df2.index.values
lc = ee.List([])
for k in indices:
    imagka = ee.Image(listaCN.get(int(k)))
    lc = lc.add(imagka)

colecao_final = ee.ImageCollection(lc)
img_15 = colecao_final.median().clipToBoundsAndScale(geometry=geometria,scale=500)
print(img_15.select(['ndvi']).getThumbUrl({'min':0, 'max':1}))

# %% Cell
mediana = colecao_final.median()
lista_sm = ee.List([])
for im in range(colecao_final.size().getInfo()):
    sub_med = ee.Image(lc.get(im)).subtract(mediana)
    lista_sm = lista_sm.add(sub_med)
    
colecao_med = ee.ImageCollection(lista_sm)

# %% Cell
defaultDummy = -99999
df_sul = pd.DataFrame()
di = {}
dias = ndvi_df2['Timestamp'].values
for j in range(colecao_med.size().getInfo()):
    tempndvi = dias[j]
    img = ee.Image(defaultDummy).blend(ee.Image(lista_sm.get(j)))
    lat250, lon250, ind250 = ext_lat_lon_pixel250(img,geometria,['ndvi'])
    di[tempndvi] = ind250[0]
    
df_sul = df_sul.from_dict(di)
df_sul = df_sul.assign(Latitude = lat250)
df_sul = df_sul.assign(Longitude = lon250)
df_sul = df_sul.set_index(['Latitude','Longitude'])
df_sul.head()

# %% Cell
oc_val = df_sul.values.reshape(-1,1)
oc_val = np.delete(oc_val, np.where(oc_val == defaultDummy))
dpoc = np.std(oc_val)
meanoc = np.mean(oc_val)
print(dpoc, meanoc)
listareg = []
alpha = 1 #float(input('digite o valor de alpha: '))
inf_lim = meanoc - alpha*dpoc
sup_lim = meanoc + alpha*dpoc
for i in oc_val:
    if inf_lim < i < sup_lim:
        listareg.append(i)
        
array_reg = np.asarray(listareg)
len(array_reg)

# %% Cell
pd.options.mode.chained_assignment = None
import sklearn
from sklearn.ensemble import IsolationForest
#treino = df_sul.values.reshape(-1,1)
#classificador = IsolationForest(n_estimators=40,random_state=0,max_samples='auto').fit(treino)

# %% Cell
def transitions(df):
    diff = df + df.shift(1)
    zero = (diff == 0).astype(int).sum()
    return zero

def mantem_normal(df):
    soma = df + df.shift(1)
    dois = (soma == 2).astype(int).sum()
    return dois

def mantem_anomalia(df):
    soma = df + df.shift(1)
    dois = (soma == -2).astype(int).sum()
    return dois

def contador_reg(df):
    return (df == 1).astype(int).sum()

def contador_anomaly(df):
    return (df == -1).astype(int).sum()

from scipy import stats
from statistics import NormalDist
def p_valor(df):
    return 1 - NormalDist().cdf(df)

# %% Cell
def isola2(df, classificador):
    x = df.to_numpy().reshape(-1,1)
    res2 = classificador.predict(x)
    #res = np.where(res2 < 0, abs(res2), 0)
    ser = pd.Series(res2, index=df.index)
    return ser

def cria_dataf(x, classificador):
    tab_ex = pd.DataFrame(index=x.index, columns= x.columns)

    for i in range(x.shape[1]):
        vals = isola2(x.iloc[:,i], classificador)
        tab_ex.iloc[:,i] = vals  
    return tab_ex

def Aplica_Metricas(x):
    tabela_de_testes = x
    tab_stats_oc = pd.DataFrame()
    tab_stats_oc.loc[:,'Anomalias'] = tabela_de_testes.apply(contador_anomaly, axis=1)
    tab_stats_oc.loc[:,'Regular'] = tabela_de_testes.apply(contador_reg, axis=1)
    tab_stats_oc.loc[:,'Mudanças'] = tabela_de_testes.apply(transitions, axis=1)
    tab_stats_oc.loc[:,'Permanece Regular'] = tabela_de_testes.apply(mantem_normal, axis=1)
    tab_stats_oc.loc[:,'Permanece Anomalia'] = tabela_de_testes.apply(mantem_anomalia, axis=1)
    ano = tab_stats_oc['Anomalias']; reg = tab_stats_oc['Regular']; mud = tab_stats_oc['Mudanças']
    media = np.mean(mud)
    stdev = np.std(mud)
    tab_stats_oc.loc[:,'media'] = media
    tab_stats_oc.loc[:,'std'] = stdev
    tab_stats_oc.loc[:,'z'] = (mud - media)/stdev
    tab_stats_oc.loc[:,'p-valor'] = tab_stats_oc.loc[:,'z'].apply(p_valor)
    return tab_stats_oc

# %% Cell
def save_tiff_fromdf(df,bands,dummy,path_out):
    
    lat = []
    lon = []
    for i in range(len(df)):
        lat.append(df.index[i][0])
        lon.append(df.index[i][1])
    
    
    ulat = np.unique(lat)
    ulon = np.unique(lon)
    ncols = len(ulon)
    nrows = len(ulat)
    nbands = len(bands)
    ys = ulat[11]-ulat[10]
    xs = ulon[11]-ulon[10]
    
    arr = np.zeros([nbands, nrows, ncols], np.float32)
    refLat = np.max(ulat)
    refLon = np.min(ulon)
    for j in range(len(df)):
        posLin = np.int64( np.round( (refLat - lat[j])/ys ) )
        posCol = np.int64( np.round( (lon[j] - refLon)/xs ) )
        for b in range(nbands):
            arr[b,posLin,posCol] = df.loc[df.index[j],bands[b]]
            
    transform = (np.min(ulon),xs,0,np.max(ulat),0,-ys)
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    
    import os
    if os.path.dirname(path_out):
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
    driver = gdal.GetDriverByName('GTiff')
    outDs = driver.Create(path_out,ncols,nrows,nbands,gdal.GDT_Float32)
    outDs.SetGeoTransform(transform)
    outDs.SetProjection(target.ExportToWkt())

    ind = 1
    for b in range(nbands):
        bandArr = np.copy(arr[b,:,:])
        outBand = outDs.GetRasterBand(ind)
        outBand.WriteArray(bandArr)
        outBand.FlushCache()
        outBand.SetNoDataValue(dummy)
        ind += 1

    outDs = None
    del outDs, outBand

    return 'ok...'

# %% Cell
data_dict_if = {}
metric_dict_if = {}
bands = ['Anomalias','p-valor']
defaultDummy = -99999
for i in range(5):
    valor = 20+ 20*i 
    texto = str(valor) + '_Estimators'
    classi = IsolationForest(n_estimators=valor,n_jobs=-1).fit(array_reg.reshape(-1,1))
    tab_ex = cria_dataf(df_sul, classi)
    data_dict_if[texto] = tab_ex
    tab_res = Aplica_Metricas(tab_ex)
    metric_dict_if[texto] = tab_res
    path_out = 'Tiff/V3/IF/Altamira_NDVI_IF_' + texto + '.tif'
    save_tiff_fromdf(tab_res,bands,defaultDummy,path_out)
    print(i)

# %% Cell
import datetime
from datetime import date

vetordate = data_dict_if['20_Estimators'].columns.to_series().dt.year.unique()
cont = 1
aux = 0
fim = vetordate[2]
col_dt_anom = {}
while (2*cont + aux) <= len(vetordate):
    x = 2*cont + aux
    fim = vetordate[x]
    comeco = vetordate[x-2]
    cont += 1
    aux += 1
    elemento = datetime.datetime.strptime(str(fim),"%Y")
    elemento1 = datetime.datetime.strptime(str(comeco),"%Y")
    limite_sup = date(elemento.year, 12, 31)
    limite_inf = date(elemento1.year, 1, 1)
    print(limite_inf, limite_sup)
    filtered = data_dict_if['20_Estimators'].loc[:,(data_dict_if['20_Estimators'].columns <= elemento)&(data_dict_if['20_Estimators'].columns >= elemento1)]
    name = str(comeco) + "-" + str(fim)
    col_dt_anom[name] = filtered
    print('ok')

# %% Cell
def apply_stats(tabela_de_testes):
    tab_stats_oc = pd.DataFrame()
    tab_stats_oc.loc[:,'Anomalias'] = tabela_de_testes.apply(contador_anomaly, axis=1)
    tab_stats_oc.loc[:,'Regular'] = tabela_de_testes.apply(contador_reg, axis=1)
    tab_stats_oc.loc[:,'Mudanças'] = tabela_de_testes.apply(transitions, axis=1)
    tab_stats_oc.loc[:,'Permanece Regular'] = tabela_de_testes.apply(mantem_normal, axis=1)
    tab_stats_oc.loc[:,'Permanece Anomalia'] = tabela_de_testes.apply(mantem_anomalia, axis=1)
    ano = tab_stats_oc['Anomalias']; reg = tab_stats_oc['Regular']; mud = tab_stats_oc['Mudanças']
    media = np.mean(mud)
    stdev = np.std(mud)
    tab_stats_oc.loc[:,'media'] = media
    tab_stats_oc.loc[:,'std'] = stdev
    tab_stats_oc.loc[:,'z'] = (mud - media)/stdev
    tab_stats_oc.loc[:,'p-valor'] = tab_stats_oc.loc[:,'z'].apply(p_valor)
    return tab_stats_oc

def monta_dict_anomaly(tab_testes):
    vetordate = tab_testes.columns.to_series().dt.year.unique()
    cont = 1
    aux = 0
    fim = vetordate[2]
    col_dt_anom = {}
    while (2*cont + aux) <= len(vetordate):
        x = 2*cont + aux
        fim = vetordate[x]
        comeco = vetordate[x-2]
        cont += 1
        aux += 1
        elemento = datetime.datetime.strptime(str(fim),"%Y")
        elemento1 = datetime.datetime.strptime(str(comeco),"%Y")
        limite_sup = date(elemento.year, 12, 31)
        limite_inf = date(elemento1.year, 1, 1)
        print(limite_inf, limite_sup)
        filtered = tab_testes.loc[:,(tab_testes.columns <= elemento)&(tab_testes.columns >= elemento1)]
        name = str(comeco) + "-" + str(fim)
        col_dt_anom[name] = filtered
        print('ok')
    return col_dt_anom

# %% Cell
chave = list(data_dict_if.keys())
bands = ['Anomalias','p-valor']
for i in chave:
    yxz = apply_stats(data_dict_if[i])
    path_out = 'Tiff/V3/IF/Valida/Alt_if_modis_' + str(i) + '.tif'
    save_tiff_fromdf(yxz,bands,defaultDummy,path_out)

# %% Cell
bands = ['Anomalias','p-valor']
for i in chave:
    w = monta_dict_anomaly(data_dict_if[i])
    for k in w.keys():
        yxz = apply_stats(w[k])
        path_out = 'Tiff/V3/IF/Valida/Alt_if_modis_' + str(i) + "_" + str(k) + '.tif'
        save_tiff_fromdf(yxz,bands,defaultDummy,path_out)

# %% Cell
def save_tiff_fromdf(df,bands,dummy,path_out):
    
    lat = []
    lon = []
    for i in range(len(df)):
        lat.append(df.index[i][0])
        lon.append(df.index[i][1])
    
    
    ulat = np.unique(lat)
    ulon = np.unique(lon)
    ncols = len(ulon)
    nrows = len(ulat)
    nbands = len(bands)
    ys = ulat[11]-ulat[10]
    xs = ulon[11]-ulon[10]
    
    arr = np.zeros([nbands, nrows, ncols], np.float32)
    refLat = np.max(ulat)
    refLon = np.min(ulon)
    for j in range(len(df)):
        posLin = np.int64( np.round( (refLat - lat[j])/ys ) )
        posCol = np.int64( np.round( (lon[j] - refLon)/xs ) )
        for b in range(nbands):
            arr[b,posLin,posCol] = df.loc[df.index[j],bands[b]]
            
    transform = (np.min(ulon),xs,0,np.max(ulat),0,-ys)
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    
    import os
    if os.path.dirname(path_out):
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
    driver = gdal.GetDriverByName('GTiff')
    outDs = driver.Create(path_out,ncols,nrows,nbands,gdal.GDT_Float32)
    outDs.SetGeoTransform(transform)
    outDs.SetProjection(target.ExportToWkt())

    ind = 1
    for b in range(nbands):
        bandArr = np.copy(arr[b,:,:])
        outBand = outDs.GetRasterBand(ind)
        outBand.WriteArray(bandArr)
        outBand.FlushCache()
        outBand.SetNoDataValue(dummy)
        ind += 1

    outDs = None
    del outDs, outBand

    return 'ok...'

# %% Cell
oc_val = df_sul.values.reshape(-1,1)
oc_val = np.delete(oc_val, np.where(oc_val == defaultDummy))
dpoc = np.std(oc_val)
meanoc = np.mean(oc_val)
print(dpoc, meanoc)
listareg = []
alpha = 0.5 #float(input('digite o valor de alpha: '))
inf_lim = meanoc - alpha*dpoc
sup_lim = meanoc + alpha*dpoc
for i in oc_val:
    if inf_lim < i < sup_lim:
        listareg.append(i)
        
array_reg = np.asarray(listareg)
len(array_reg)

# %% Cell
beta = 0.005 #float(input('digite o valor de beta: '))
novo_data = int(beta * len(array_reg))
dataind = np.random.choice(len(array_reg),size=novo_data)
dataset = array_reg[dataind]
len(dataset)

# %% Cell
pd.options.mode.chained_assignment = None
import sklearn
from sklearn.svm import OneClassSVM

occlf = OneClassSVM(nu=0.05).fit(dataset.reshape(-1,1))

def OCSVM(df):
    y = df.to_numpy().reshape(-1,1)
    res = occlf.predict(y)
    ser = pd.Series(res, index=df.index)
    return ser

# %% Cell
def isola2(df, classificador):
    x = df.to_numpy().reshape(-1,1)
    res2 = classificador.predict(x)
    #res = np.where(res2 < 0, abs(res2), 0)
    ser = pd.Series(res2, index=df.index)
    return ser

def cria_dataf(x, classificador):
    tab_ex = pd.DataFrame(index=x.index, columns= x.columns)

    for i in range(x.shape[1]):
        vals = isola2(x.iloc[:,i], classificador)
        tab_ex.iloc[:,i] = vals  
    return tab_ex

# %% Cell
def Aplica_Metricas(x):
    tabela_de_testes = x
    tab_stats_oc = pd.DataFrame()
    tab_stats_oc.loc[:,'Anomalias'] = tabela_de_testes.apply(contador_anomaly, axis=1)
    tab_stats_oc.loc[:,'Regular'] = tabela_de_testes.apply(contador_reg, axis=1)
    tab_stats_oc.loc[:,'Mudanças'] = tabela_de_testes.apply(transitions, axis=1)
    tab_stats_oc.loc[:,'Permanece Regular'] = tabela_de_testes.apply(mantem_normal, axis=1)
    tab_stats_oc.loc[:,'Permanece Anomalia'] = tabela_de_testes.apply(mantem_anomalia, axis=1)
    ano = tab_stats_oc['Anomalias']; reg = tab_stats_oc['Regular']; mud = tab_stats_oc['Mudanças']
    media = np.mean(mud)
    stdev = np.std(mud)
    tab_stats_oc.loc[:,'media'] = media
    tab_stats_oc.loc[:,'std'] = stdev
    tab_stats_oc.loc[:,'z'] = (mud - media)/stdev
    tab_stats_oc.loc[:,'p-valor'] = tab_stats_oc.loc[:,'z'].apply(p_valor)
    return tab_stats_oc

# %% Cell
def transitions(df):
    diff = df + df.shift(1)
    zero = (diff == 0).astype(int).sum()
    return zero

def mantem_normal(df):
    soma = df + df.shift(1)
    dois = (soma == 2).astype(int).sum()
    return dois

def mantem_anomalia(df):
    soma = df + df.shift(1)
    dois = (soma == -2).astype(int).sum()
    return dois

def contador_reg(df):
    return (df == 1).astype(int).sum()

def contador_anomaly(df):
    return (df == -1).astype(int).sum()

from scipy import stats
from statistics import NormalDist
def p_valor(df):
    return 1 - NormalDist().cdf(df)

# %% Cell
nuu = [0.025, 0.05, 0.1]
data_dict_oc = {}
metric_dict = {}
bands = ['Anomalias','p-valor']
for i in nuu:
    classi = OneClassSVM(nu=i, kernel='rbf', gamma='auto').fit(dataset.reshape(-1,1))
    tab_ex = cria_dataf(df_sul, classi)
    data_dict_oc[i] = tab_ex
    tab_res = Aplica_Metricas(tab_ex)
    metric_dict[i] = tab_res
    path_out = 'Tiff/V3/Altamira_NDVI_OCSVM_' + str(i) + '.tif'
    save_tiff_fromdf(tab_res,bands,defaultDummy,path_out)
    print(i)

# %% Cell
import matplotlib.pyplot as plt
mask1 = tab_res['Anomalias'].values
fig, ax = plt.subplots(figsize=(10,8))
ax.scatter(x,y,c=mask1,cmap='turbo')
ax.set_title('Anomalias')
plt.colorbar(ax.scatter(x,y,c=mask1,cmap='turbo'),ax=ax)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.savefig('Mari30_NDVI_OC_anom.png')
plt.show()

# %% Cell
import time
import datetime

elemento = datetime.datetime.strptime("2016","%Y")
elemento1 = datetime.datetime.strptime("2013","%Y")

# %% Cell
data_dict_oc[0.025]

# %% Cell
data_dict_oc[0.025].columns.to_series().dt.year.unique()

# %% Cell
from datetime import date

# %% Cell
vetordate = data_dict_oc[0.025].columns.to_series().dt.year.unique()
cont = 1
aux = 0
fim = vetordate[2]
col_dt_anom = {}
while (2*cont + aux) <= len(vetordate):
    x = 2*cont + aux
    fim = vetordate[x]
    comeco = vetordate[x-2]
    cont += 1
    aux += 1
    elemento = datetime.datetime.strptime(str(fim),"%Y")
    elemento1 = datetime.datetime.strptime(str(comeco),"%Y")
    limite_sup = date(elemento.year, 12, 31)
    limite_inf = date(elemento1.year, 1, 1)
    print(limite_inf, limite_sup)
    filtered = data_dict_oc[0.025].loc[:,(data_dict_oc[0.025].columns <= elemento)&(data_dict_oc[0.025].columns >= elemento1)]
    name = str(comeco) + "-" + str(fim)
    col_dt_anom[name] = filtered
    print('ok')

# %% Cell
chave = list(data_dict_oc.keys())

# %% Cell
def apply_stats(tabela_de_testes):
    tab_stats_oc = pd.DataFrame()
    tab_stats_oc.loc[:,'Anomalias'] = tabela_de_testes.apply(contador_anomaly, axis=1)
    tab_stats_oc.loc[:,'Regular'] = tabela_de_testes.apply(contador_reg, axis=1)
    tab_stats_oc.loc[:,'Mudanças'] = tabela_de_testes.apply(transitions, axis=1)
    tab_stats_oc.loc[:,'Permanece Regular'] = tabela_de_testes.apply(mantem_normal, axis=1)
    tab_stats_oc.loc[:,'Permanece Anomalia'] = tabela_de_testes.apply(mantem_anomalia, axis=1)
    ano = tab_stats_oc['Anomalias']; reg = tab_stats_oc['Regular']; mud = tab_stats_oc['Mudanças']
    media = np.mean(mud)
    stdev = np.std(mud)
    tab_stats_oc.loc[:,'media'] = media
    tab_stats_oc.loc[:,'std'] = stdev
    tab_stats_oc.loc[:,'z'] = (mud - media)/stdev
    tab_stats_oc.loc[:,'p-valor'] = tab_stats_oc.loc[:,'z'].apply(p_valor)
    return tab_stats_oc

# %% Cell
def monta_dict_anomaly(tab_testes):
    vetordate = tab_testes.columns.to_series().dt.year.unique()
    cont = 1
    aux = 0
    fim = vetordate[2]
    col_dt_anom = {}
    while (2*cont + aux) <= len(vetordate):
        x = 2*cont + aux
        fim = vetordate[x]
        comeco = vetordate[x-2]
        cont += 1
        aux += 1
        elemento = datetime.datetime.strptime(str(fim),"%Y")
        elemento1 = datetime.datetime.strptime(str(comeco),"%Y")
        limite_sup = date(elemento.year, 12, 31)
        limite_inf = date(elemento1.year, 1, 1)
        print(limite_inf, limite_sup)
        filtered = tab_testes.loc[:,(tab_testes.columns <= elemento)&(tab_testes.columns >= elemento1)]
        name = str(comeco) + "-" + str(fim)
        col_dt_anom[name] = filtered
        print('ok')
    return col_dt_anom

# %% Cell
## One-Class SVM
bands = ['Anomalias','p-valor']
for i in chave:
    yxz = apply_stats(data_dict_oc[i])
    path_out = 'Tiff/V3//Alt_OCSVM_modis_' + str(i) + '.tif'
    save_tiff_fromdf(yxz,bands,defaultDummy,path_out)

# %% Cell
cole_dt_anom_IF = monta_dict_anomaly(tabela_de_testes)
cole_dt_anom_IF

# %% Cell
bands = ['Anomalias','p-valor']
for i in chave:
    w = monta_dict_anomaly(data_dict_oc[i])
    for k in w.keys():
        yxz = apply_stats(w[k])
        path_out = 'Tiff/V3//Alt_OCSVM_nu_' + str(i) + "_" + str(k) + '.tif'
        save_tiff_fromdf(yxz,bands,defaultDummy,path_out)

# %% Cell
datadia = '2020-07-26'
ref_date = datetime.datetime.strptime(datadia, "%Y-%m-%d")

res = min(data_dict_oc[0.025].columns, key=lambda sub: abs(sub - ref_date))
res

# %% Cell
type(res) == type(data_dict_oc[0.025].columns[0])

# %% Cell
str(res)[:10]

# %% Cell
for i in chave:
    for item in data_dict_oc[i].columns:
        if res == item:
            print(str(item)[:10])
            downdata_if = data_dict_oc[i].loc[:,:item]
            path_out = 'Tiff/V3//Alt_OCSVM_modis_' + str(item)[:10] + '.tif'
            save_tiff_fromdf(downdata_if,[item],defaultDummy,path_out)
            #downdata_oc = tabela_de_testes.loc[:,:item]
            #path_out = 'Tiff/Alt_OCSVM_modis_' + str(item)[:10] + '.tif'
            #save_tiff_fromdf(downdata_oc,[item],defaultDummy,path_out)

# %% Cell
data_dict_oc[0.025].loc[:]

# %% Cell
col_dt_anom

# %% Cell

