#%%
import pandas as pd
import numpy as np



# %%

cols = ['id','amount_tsh','date_recorded','funder','gps_height','installer','longitude','latitude','wpt_name','num_private','basin','subvillage','region','region_code','district_code','lga','ward','population','public_meeting','recorded_by','scheme_management','scheme_name','permit','construction_year','extraction_type','extraction_type_group','extraction_type_class','management','management_group','payment','payment_type','water_quality','quality_group','quantity','quantity_group','source','source_type','source_class','waterpoint_type','waterpoint_type_group']
#cols = [str(cols.index(col)) for col in cols]
trainData = pd.read_csv('piu_train.csv', names=cols, sep=',', header=0)


# %%
