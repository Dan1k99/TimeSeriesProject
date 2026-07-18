
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
coordenadas = "-44.13576882693976,-20.14040952973489,-44.107444699742494,-20.11204199216084"
x1,y1,x2,y2 = coordenadas.split(",")
geometria = geometry = ee.Geometry.Polygon(
        [[[float(x1),float(y2)],
          [float(x2),float(y2)],
          [float(x2),float(y1)],
          [float(x1),float(y1)],
          [float(x1),float(y2)]]])
datas = "2013-01-01,2021-12-31"
inicio,fim = datas.split(",")
colecao = ee.ImageCollection('COPERNICUS/S2').filterBounds(geometria).filterDate(inicio,fim).filterMetadata('CLOUD_COVERAGE_ASSESSMENT','less_than', 0.5)
print("Total de imagens encontradas: "+str(colecao.size().getInfo()))

# %% Cell
blue = 'B2'; green = 'B3'; red = 'B4'; nir = 'B8'

def ndvi(imagem):
    ndvi = imagem.expression('(nir - red) / (nir + red)',{'nir':imagem.select(nir),'red':imagem.select(red)}).rename('ndvi')
    return imagem.addBands(ndvi)

def ndwi(imagem):
    ndwi = imagem.expression('(nir - green) / (nir + green)',{'nir':imagem.select(nir),'green':imagem.select(green)}).rename('ndwi')
    return imagem.addBands(ndwi)

colecao = colecao.map(ndvi)
colecao = colecao.map(ndwi)
imagem = colecao.median()
print(imagem.bandNames().getInfo())

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
reduce_ndvi = create_reduce_region_function(
    geometry=geometria, reducer=ee.Reducer.mean(), scale=10)

ndvi_stat_fc = ee.FeatureCollection(colecao.map(reduce_ndvi)).filter(
    ee.Filter.notNull(colecao.first().bandNames()))

ndvi_dict = fc_to_dict(ndvi_stat_fc).getInfo()
ndvi_df = pd.DataFrame(ndvi_dict)
display(ndvi_df)
print(ndvi_df.dtypes)

# %% Cell
ndvi_df = add_date_info(ndvi_df)
ndvi_df.tail(5)

# %% Cell
import altair as alt
highlight = alt.selection(
    type='single', on='mouseover', fields=['Year'], nearest=True)

base = alt.Chart(ndvi_df).encode(
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
base = alt.Chart(ndvi_df).encode(
    x=alt.X('DOY:Q', scale=alt.Scale(domain=(0, 360))))

line = base.mark_line().encode(
    y=alt.Y('median(ndvi):Q', scale=alt.Scale(domain=(0, 1))))

band = base.mark_errorband(extent='iqr').encode(
    y='ndvi:Q')

(line + band).properties(width=600, height=300).interactive()

# %% Cell
listaColecao = colecao.toList(colecao.size())
lista_sm = ee.List([])
for im in range(colecao.size().getInfo()):
    sub_med = ee.Image(listaColecao.get(im)).subtract(imagem)
    lista_sm = lista_sm.add(sub_med)
    
col_ajust = ee.ImageCollection(lista_sm)

# %% Cell
def ext_lat_lon_pixel10(image, geometria, bandas):
    image = image.addBands(ee.Image.pixelLonLat())
    coordenadas = image.select(['longitude', 'latitude']+bandas).reduceRegion(reducer=ee.Reducer.toList(),geometry=geometria,scale=10,bestEffort=True)
    bandas_valores = []
    for banda in bandas:
        bandas_valores.append(np.array(ee.List(coordenadas.get(banda)).getInfo()).astype(float))

    return np.array(ee.List(coordenadas.get('latitude')).getInfo()).astype(float), np.array(ee.List(coordenadas.get('longitude')).getInfo()).astype(float), bandas_valores

# %% Cell
listaCN = col_ajust.toList(col_ajust.size())
defaultDummy = -99999
df_r10 = pd.DataFrame()
dias = ndvi_df['Timestamp'].values
di = {}
for j in range(col_ajust.size().getInfo()):
    tempndvi = dias[j]
    img = ee.Image(defaultDummy).blend(ee.Image(listaCN.get(j)))
    lat10, lon10, ind10 = ext_lat_lon_pixel10(img,geometria,['ndwi'])
    di[tempndvi] = ind10[0]
    
df_r10 = df_r10.from_dict(di)
df_r10 = df_r10.assign(Latitude = lat10)
df_r10 = df_r10.assign(Longitude = lon10)
df_r10 = df_r10.set_index(['Latitude','Longitude'])
df_r10

# %% Cell
oc_val = df_r10.values.reshape(-1,1)
dpoc = np.std(oc_val)
meanoc = np.mean(oc_val)
print(dpoc, meanoc)

# %% Cell
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
beta = 0.01 #float(input('digite o valor de beta: '))
novo_data = int(beta * len(array_reg))
dataind = np.random.choice(len(array_reg),size=novo_data)
dataset = array_reg[dataind]
len(dataset)

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
pd.options.mode.chained_assignment = None
import sklearn
from sklearn.svm import OneClassSVM

def OCSVM(df, occlf):
    y = df.to_numpy().reshape(-1,1)
    res = occlf.predict(y)
    ser = pd.Series(res, index=df.index)
    return ser

def cria_dataf(x, classificador):
    tab_ex = pd.DataFrame(index=x.index, columns= x.columns)

    for i in range(x.shape[1]):
        vals = OCSVM(x.iloc[:,i], classificador)
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
nuu = [0.001, 0.005, 0.01, 0.05, 0.1]
metric_dict = {}
data_dict = {}
dummy = -99999
bands = ['Anomalias','p-valor']
for i in nuu:
    classificador = OneClassSVM(nu=i, kernel='rbf', gamma='auto').fit(array_reg)
    tab_ex = cria_dataf(df_r10, classificador)
    data_dict[i] = tab_ex
    tab_res = Aplica_Metricas(tab_ex)
    metric_dict[i] = tab_res
    path_out = 'Tiff/V4_Out_22/Brumadinho_NDWI_OCSVM_' + str(i) + '.tif'
    save_tiff_fromdf(tab_res,bands,dummy,path_out)
    print(i)

# %% Cell
import matplotlib.pyplot as plt
filtro_oc = tab_stats_oc.copy()
filtro_oc.loc[tab_stats_oc['p-valor']>0.05,'p-valor'] = 0
filtro_oc.loc[tab_stats_oc['p-valor']<0.05,'p-valor'] = 1
x = filtro_oc.index.get_level_values(1).values
y = filtro_oc.index.get_level_values(0).values

mask1 = filtro_oc['p-valor'].values
fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(x,y,c=mask1,cmap='binary')
ax.set_title('P-Valores menores que 5%')
plt.colorbar(ax.scatter(x,y,c=mask1,marker=',',cmap='binary'),ax=ax)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.savefig('Mari30_NDVI_OC_pval.png')
plt.show()

# %% Cell
mask1 = tab_stats_oc['Anomalias'].values
fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(x,y,c=mask1,cmap='turbo')
ax.set_title('Anomalias')
plt.colorbar(ax.scatter(x,y,c=mask1,cmap='turbo'),ax=ax)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.savefig('Mari30_NDVI_OC_anom.png')
plt.show()

# %% Cell
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def isola(df, occlf):
    y = df.to_numpy().reshape(-1,1)
    res = occlf.predict(y)
    ser = pd.Series(res, index=df.index)
    return ser

def cria_dataf_2(x, classificador):
    tab_ex = pd.DataFrame(index=x.index, columns= x.columns)

    for i in range(x.shape[1]):
        vals = isola(x.iloc[:,i], classificador)
        tab_ex.iloc[:,i] = vals  
    return tab_ex

# %% Cell
data_dict_if = {}
metric_dict_if = {}
bands = ['Anomalias','p-valor']
dummy = -99999
for i in range(5):
    valor = 20 + i*20
    classificador = IsolationForest(n_estimators=valor,n_jobs=-1).fit(array_reg)
    
    tab_ex = cria_dataf_2(df_r10, classificador)
    data_dict_if[i] = tab_ex
    tab_res = Aplica_Metricas(tab_ex)
    metric_dict_if[i] = tab_res
    path_out = 'Tiff/V4_Out_22/Brumadinho_NDWI_IF_' + str(valor) + '.tif'
    save_tiff_fromdf(tab_res,bands,dummy,path_out)
    print(i)

# %% Cell
import matplotlib.pyplot as plt
filtro = tab_stats.copy()
filtro.loc[tab_stats['p-valor']>0.05,'p-valor'] = 0
filtro.loc[tab_stats['p-valor']<0.05,'p-valor'] = 1
y = filtro.index.get_level_values(0).values
x = filtro.index.get_level_values(1).values
mask1 = filtro['p-valor'].values
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x,y,c=mask1,marker=',',cmap='binary')
ax.set_title('Isolation Forest p-values mask')
plt.colorbar(ax.scatter(x,y,c=mask1,cmap='binary'),ax=ax)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.savefig('Bru10_NDVIIFpvalues.png')
plt.show()

# %% Cell
mask = tab_stats['Anomalias'].values
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x,y,c=mask,marker=',',cmap='turbo')
ax.set_title('Anomalias')
plt.colorbar(ax.scatter(x,y,c=mask,marker=',',cmap='turbo'),ax=ax)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.savefig('Bru10_NDVIIFanomalias.png')
plt.show()

# %% Cell
classificador2 = IsolationForest(n_estimators=40,random_state=0,
                                n_jobs=-1,max_samples='auto').fit(oc_val)
def isola2(df):
    x = df.to_numpy().reshape(-1,1)
    res = classificador2.predict(x)
    ser = pd.Series(res, index=df.index)
    return ser

# %% Cell
tabela_de_testes_if = pd.DataFrame(index = df_r10.index, columns = df_r10.columns)
tabela_de_testes_if = tabela_de_testes_if.append(df_r10.apply(isola2, axis=1), ignore_index=False).dropna()
tabela_de_testes_if.tail()

# %% Cell
tab_stats_if = pd.DataFrame()
tab_stats_if.loc[:,'Anomalias'] = tabela_de_testes_if.apply(contador_anomaly, axis=1)
tab_stats_if.loc[:,'Regular'] = tabela_de_testes_if.apply(contador_reg, axis=1)
tab_stats_if.loc[:,'Mudanças'] = tabela_de_testes_if.apply(transitions, axis=1)
tab_stats_if.loc[:,'Permanece Regular'] = tabela_de_testes_if.apply(mantem_normal, axis=1)
tab_stats_if.loc[:,'Permanece Anomalia'] = tabela_de_testes_if.apply(mantem_anomalia, axis=1)
ano = tab_stats_if['Anomalias']; reg = tab_stats_if['Regular']; mud = tab_stats_if['Mudanças']
media = np.mean(mud)
stdev = np.std(mud)
tab_stats_if.loc[:,'media'] = media
tab_stats_if.loc[:,'std'] = stdev
tab_stats_if.loc[:,'z'] = (mud - media)/stdev
tab_stats_if.loc[:,'p-valor'] = tab_stats_if.loc[:,'z'].apply(p_valor)
tab_stats_if

# %% Cell
filtro = tab_stats_if.copy()
filtro.loc[tab_stats_if['p-valor']>0.05,'p-valor'] = 0
filtro.loc[tab_stats_if['p-valor']<0.05,'p-valor'] = 1
y = filtro.index.get_level_values(0).values
x = filtro.index.get_level_values(1).values
mask1 = filtro['p-valor'].values
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x,y,c=mask1,marker=',',cmap='binary')
ax.set_title('Isolation Forest p-values mask')
plt.colorbar(ax.scatter(x,y,c=mask1,cmap='binary'),ax=ax)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.savefig('Bru10_NDVIIFpvalues.png')
plt.show()

# %% Cell
mask = tab_stats_if['Anomalias'].values
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x,y,c=mask,marker=',',cmap='turbo')
ax.set_title('Anomalias')
plt.colorbar(ax.scatter(x,y,c=mask,marker=',',cmap='turbo'),ax=ax)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.savefig('Bru10_NDVIIFanomalias.png')
plt.show()

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
#precisa rodar em 15/06/2022
path_out = 'Tiff/V2_Jun22/Brumadinho_NDVI_IFAnomalies.tif'
bands = ['Anomalias','p-valor']
save_tiff_fromdf(tab_stats_if,bands,defaultDummy,path_out)
# save_tiff_fromdf(tab_stats_oc,bands,defaultDummy,path_out)

# %% Cell
import altair as alt
highlight = alt.selection(
    type='single', on='mouseover', fields=['Year'], nearest=True)

base = alt.Chart(ndvi_df).encode(
    x=alt.X('DOY:Q', scale=alt.Scale(domain=[0, 353], clamp=True)),
    y=alt.Y('ndwi:Q', scale=alt.Scale(domain=[0, 1])),
    color=alt.Color('Year:O', scale=alt.Scale(scheme='magma')))

points = base.mark_circle().encode(
    opacity=alt.value(0),
    tooltip=[
        alt.Tooltip('Year:O', title='Year'),
        alt.Tooltip('DOY:Q', title='DOY'),
        alt.Tooltip('ndwi:Q', title='NDVI')
    ]).add_selection(highlight)

lines = base.mark_line().encode(
    size=alt.condition(~highlight, alt.value(1), alt.value(3)))

(points + lines).properties(width=600, height=350).interactive()

# %% Cell
base = alt.Chart(ndvi_df).encode(
    x=alt.X('DOY:Q', scale=alt.Scale(domain=(0, 360))))

line = base.mark_line().encode(
    y=alt.Y('median(ndvi):Q', scale=alt.Scale(domain=(0, 1))))

band = base.mark_errorband(extent='iqr').encode(
    y='ndvi:Q')

(line + band).properties(width=600, height=300).interactive()

# %% Cell
listaCN = col_ajust.toList(col_ajust.size())
defaultDummy = -99999
df_ndwi = pd.DataFrame()
dias = ndvi_df['Timestamp'].values
di = {}
for j in range(col_ajust.size().getInfo()):
    tempndwi = dias[j]
    img = ee.Image(defaultDummy).blend(ee.Image(listaCN.get(j)))
    lat10, lon10, ind10 = ext_lat_lon_pixel10(img,geometria,['ndwi'])
    di[tempndwi] = ind10[0]
    
df_ndwi = df_ndwi.from_dict(di)
df_ndwi = df_ndwi.assign(Latitude = lat10)
df_ndwi = df_ndwi.assign(Longitude = lon10)
df_ndwi = df_ndwi.set_index(['Latitude','Longitude'])
df_ndwi

# %% Cell
pd.options.mode.chained_assignment = None
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
train = df_ndwi.values.reshape(-1,1)
classificador = IsolationForest(n_estimators=40,random_state=0,
                                n_jobs=-1,max_samples='auto').fit(train)

# %% Cell
def isola(df):
    x = df.to_numpy().reshape(-1,1)
    res = classificador.predict(x)
    ser = pd.Series(res, index=df.index)
    return ser

# %% Cell
ndwi_testes = pd.DataFrame(index = df_ndwi.index, columns = df_ndwi.columns)
ndwi_testes = ndwi_testes.append(df_ndwi.apply(isola, axis=1), ignore_index=False).dropna()
ndwi_testes.tail()

# %% Cell
ndwi_stats2 = pd.DataFrame()
ndwi_stats2.loc[:,'Anomalias'] = ndwi_testes.apply(contador_anomaly, axis=1)
ndwi_stats2.loc[:,'Regular'] = ndwi_testes.apply(contador_reg, axis=1)
ndwi_stats2.loc[:,'Mudanças'] = ndwi_testes.apply(transitions, axis=1)
ndwi_stats2.loc[:,'Permanece Regular'] = ndwi_testes.apply(mantem_normal, axis=1)
ndwi_stats2.loc[:,'Permanece Anomalia'] = ndwi_testes.apply(mantem_anomalia, axis=1)
ano = ndwi_stats2['Anomalias']; reg = ndwi_stats2['Regular']; mud = ndwi_stats2['Mudanças']
media = np.mean(mud)
stdev = np.std(mud)
ndwi_stats2.loc[:,'media'] = media
ndwi_stats2.loc[:,'std'] = stdev
ndwi_stats2.loc[:,'z'] = (mud - media)/stdev
ndwi_stats2.loc[:,'p-valor'] = ndwi_stats2.loc[:,'z'].apply(p_valor)
ndwi_stats2

# %% Cell
import matplotlib.pyplot as plt
filtrondwi = ndwi_stats2.copy()
filtrondwi.loc[ndwi_stats2['p-valor']>0.05,'p-valor'] = 0
filtrondwi.loc[ndwi_stats2['p-valor']<0.05,'p-valor'] = 1
y = filtrondwi.index.get_level_values(0).values
x = filtrondwi.index.get_level_values(1).values
mask1 = filtrondwi['p-valor'].values
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x,y,c=mask1,marker=',',cmap='binary')
ax.set_title('NDWI Isolation Forest p-values mask')
plt.colorbar(ax.scatter(x,y,c=mask1,cmap='binary'),ax=ax)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.savefig('Bru10_NDWIIFpvalues.png')
plt.show()

# %% Cell
mask = ndwi_stats2['Anomalias'].values
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x,y,c=mask,marker=',',cmap='turbo')
ax.set_title('Anomalias NDWI')
plt.colorbar(ax.scatter(x,y,c=mask,marker=',',cmap='turbo'),ax=ax)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.savefig('Bru10_NDWIIFanomalias.png')
plt.show()

# %% Cell
path_out = 'Tiff/V2_Jun22/Bru_NDWI_IF_sentinel.tif'
bands = ['Anomalias','p-valor']
save_tiff_fromdf(ndwi_stats2,bands,defaultDummy,path_out)

# %% Cell
oc_ndwi = df_ndwi.values.reshape(-1,1)
dpoc = np.std(oc_ndwi)
meanoc = np.mean(oc_ndwi)
print(dpoc, meanoc)

# %% Cell
ndwireg = []
alpha = float(input('digite o valor de α: '))
inf_lim = meanoc - alpha*dpoc
sup_lim = meanoc + alpha*dpoc
for i in oc_ndwi:
    if inf_lim < i < sup_lim:
        ndwireg.append(i)
        
arrndwi_reg = np.asarray(ndwireg)
len(arrndwi_reg)

# %% Cell
novo_data = int(0.02 * len(arrndwi_reg))
dataind = np.random.choice(len(arrndwi_reg),size=novo_data)
ndwi_data = arrndwi_reg[dataind]
len(ndwi_data)

# %% Cell
pd.options.mode.chained_assignment = None
import sklearn
from sklearn.svm import OneClassSVM
occlf = OneClassSVM(nu=0.05).fit(ndwi_data)

def OCSVM2(df):
    y = df.to_numpy().reshape(-1,1)
    res = occlf.predict(y)
    ser = pd.Series(res, index=df.index)
    return ser

# %% Cell
ndwi_teste = pd.DataFrame(index = df_ndwi.index, columns = df_ndwi.columns)
ndwi_teste = ndwi_teste.append(df_ndwi.apply(OCSVM2, axis=1), ignore_index=False).dropna()
ndwi_teste.tail()

# %% Cell
ndwi_octab = pd.DataFrame()
ndwi_octab.loc[:,'Anomalias'] = ndwi_teste.apply(contador_anomaly, axis=1)
ndwi_octab.loc[:,'Regular'] = ndwi_teste.apply(contador_reg, axis=1)
ndwi_octab.loc[:,'Mudanças'] = ndwi_teste.apply(transitions, axis=1)
ndwi_octab.loc[:,'Permanece Regular'] = ndwi_teste.apply(mantem_normal, axis=1)
ndwi_octab.loc[:,'Permanece Anomalia'] = ndwi_teste.apply(mantem_anomalia, axis=1)
ano = ndwi_octab['Anomalias']; reg = ndwi_octab['Regular']; mud = ndwi_octab['Mudanças']
media = np.mean(mud)
stdev = np.std(mud)
ndwi_octab.loc[:,'media'] = media
ndwi_octab.loc[:,'std'] = stdev
ndwi_octab.loc[:,'z'] = (mud - media)/stdev
ndwi_octab.loc[:,'p-valor'] = ndwi_octab.loc[:,'z'].apply(p_valor)
ndwi_octab

# %% Cell
ft_ndwi_oc = ndwi_octab.copy()
ft_ndwi_oc.loc[ndwi_octab['p-valor']>0.05,'p-valor'] = 0
ft_ndwi_oc.loc[ndwi_octab['p-valor']<0.05,'p-valor'] = 1
x = ft_ndwi_oc.index.get_level_values(1).values
y = ft_ndwi_oc.index.get_level_values(0).values

mask2 = ft_ndwi_oc['p-valor'].values
fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(x,y,c=mask2,cmap='binary')
ax.set_title('One-Class SVM p-values mask')
plt.colorbar(ax.scatter(x,y,c=mask2,marker=',',s=2,cmap='binary'),ax=ax)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.savefig('Mari30_NDVI_IF_pval.png')
plt.show()

# %% Cell
mask3 = ndwi_octab['Anomalias'].values
fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(x,y,c=mask3,marker=',',cmap='turbo')
ax.set_title('One-Class SVM 2013-2020 Anomaly Detection')
plt.colorbar(ax.scatter(x,y,c=mask3,marker=',',cmap='turbo'),ax=ax)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.savefig('Mari30__NDVI_IF_anom.png')
plt.show()

# %% Cell
path_out = 'Tiff//Bru_NDWI_OC_sentinel.tif'
bands = ['Anomalias','p-valor']
save_tiff_fromdf(ft_ndwi_oc,bands,defaultDummy,path_out)

# %% Cell
df_r10.columns.values
#O INTERESSE SÃO DATAS PRÓXIMAS AO DIA 25-01-2019

# %% Cell
tabela_de_testes.iloc[:,15]

# %% Cell
pre_romp = tabela_de_testes.iloc[:,15].values
fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(x,y,c=pre_romp,marker=',',cmap='binary')
ax.set_title('Isolation Forest 2018-07-21 Anomaly Detection')
plt.colorbar(ax.scatter(x,y,c=pre_romp,marker=',',cmap='binary'),ax=ax)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#plt.savefig('Bru30_IFanomalias.png')
plt.show()

# %% Cell

