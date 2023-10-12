import rasterio as rio
import numpy as np
import sys
import os

def normalize(data):
    min_val = np.percentile(data,2)
    max_val = np.percentile(data,98)
    new_data = (data - min_val ) / (max_val - min_val)
    return np.clip(new_data,0,1)

def getData(fileName):
    src = rio.open(fileName)
    band = src.read()
    src.close()
    return np.moveaxis(band, (0,1,2),(2,0,1) )

def extractData(rs_dir, suffix, year):
    print("working on %d"%year)
    prefix = "s2_%d"%year
    gt_fileName = "gt_data_%d.npy"%year
    gt_data = np.load(gt_fileName)
    gt_data_idx = gt_data[:,0:2].astype("int")
    gt_data_idx = (gt_data_idx[:,0], gt_data_idx[:,1])
    full_data = []
    for s in suffix:
        fileName = rs_dir+"/"+prefix+"_"+s+".tif"
        band_ts = getData(fileName)
        band_ts = normalize(band_ts)
        temp = band_ts[ gt_data_idx ]
        full_data.append(temp)
    full_data = np.stack(full_data,axis=1)
    np.save("data_%d.npy"%year,full_data)


rs_dir = "/home/edgar/DATA/Sentinel2images/old"
suffix = ["B02","B03","B04","B05","B06","B07","B08","B11","B12","B8A"]


#for year in [2018,2020,2021]:
#for year in [2018,2021]:
for year in [2020]:
    extractData(rs_dir, suffix, year)

