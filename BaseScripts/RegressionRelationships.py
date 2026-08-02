import os
import numpy as np
import pandas as pd
import glob as glob
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import calendar

os.chdir('/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/HOURLY_DAILY/REGIONAL/REG_SHP')
fl = gpd.read_file('New_regions_IJOC_2019.shp')

# ax=plt.subplot(111)
# fl.plot(ax=ax)

META = pd.read_csv('/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/HOURLY_DAILY/META_HUK_GAUGE.csv')
META_GDF = gpd.GeoDataFrame(META,geometry=gpd.points_from_xy(META['longitude'],META['latitude']),crs='EPSG:4326')
fl = fl.to_crs("EPSG:4326")

META_REGION = gpd.sjoin(META_GDF,fl[['name','geometry']],how='left',predicate='within')
META_REGION = META_REGION.rename(columns={'name':'Region'})
META_REGION = META_REGION.drop(columns=['index_right'])
META_REGION = pd.DataFrame(META_REGION)
META_REGION = META_REGION.drop('geometry',axis=1) 

missing = META_REGION[META_REGION['Region'].isna()].copy()
missing = gpd.GeoDataFrame(missing,geometry=gpd.points_from_xy(missing['longitude'],missing['latitude']),crs='EPSG:4326')
fl = fl.to_crs("EPSG:4326")
nearest = gpd.sjoin_nearest(missing,fl[['name','geometry']],how='left',distance_col='distance_deg')
META_REGION.loc[META_REGION['Region'].isna(),'Region'] = nearest['name'].values

META_REGION = gpd.GeoDataFrame(META_REGION,geometry=gpd.points_from_xy(META_REGION['longitude'],META_REGION['latitude']),crs='EPSG:4326')
# fl = fl.to_crs('EPSG:4326')
# REG_COUNT = (META_REGION['Region'].value_counts().sort_index())
# fig, ax = plt.subplots(figsize=(9,11))

# # Region polygons
# fl.plot(ax=ax,column='name',cmap='Pastel1',alpha=0.4,edgecolor='black',linewidth=1)
# colors = plt.cm.Set1.colors
# for i, reg in enumerate(sorted(META_REGION['Region'].unique())):
#     subset = META_REGION[META_REGION['Region'] == reg]
#     subset.plot(ax=ax,markersize=20,color=colors[i],marker='o',label=f"{reg} ({REG_COUNT[reg]})")

# ax.legend(loc='upper left')
# ax.set_title('Gauges in Extreme Rainfall regions',fontsize=14,weight='bold')
# # ax.set_xlabel('Longitude')
# # ax.set_ylabel('Latitude')
# plt.tight_layout()
# plt.savefig('/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/PaperPlots/SuppF1.png', dpi=600, bbox_inches='tight')
# plt.show()

STNS = [META_REGION['FILENAME'][META_REGION['Region']==u].values for u in ['A','B','C','D','E']]



os.chdir('/home/users/REF_STAT_RF_HR_V2')
fig, axes = plt.subplots(3, 5,figsize=(28,18),sharex=False,sharey=False)
region_names = ['A','B','C','D','E']

for r in range(5):

    REF = []
    for f in ('REF_' + STNS[r]):
        REF.append(pd.read_parquet(f))

    vardf = pd.concat([x[x['name']=='variance'] for x in REF],ignore_index=True)
    var24 = vardf[vardf.duration=='24H']['value'].values
    var1  = vardf[vardf.duration=='1H']['value'].values
    Xv = np.log(var24)
    Yv = np.log(var1)
    slope, intercept, rval, _, _ = linregress(Xv,Yv)
    xx = np.linspace(Xv.min(),Xv.max(),100)
    ax = axes[0,r]
    ax.scatter(Xv,Yv,s=10,alpha=0.65)
    ax.plot(xx,intercept+slope*xx,color='red',lw=2)
    ax.set_title(f"Region {region_names[r]}",fontsize=16,fontweight='bold')
    ax.text(0.03,0.97,f"ln(V$_{{1H}}$) = {intercept:.3f}" f" + {slope:.3f} ln(V$_{{24H}}$)\n\n" f"R$^2$ = {rval**2:.3f}",transform=ax.transAxes,fontsize=11,fontweight='bold',va='top')

    if r==0:
        ax.set_ylabel("ln(Variance 1H)",fontsize=14,fontweight='bold')
        
    ax.set_xlabel("ln(Variance 24H)",fontsize=13,fontweight='bold')

   

    skewdf = pd.concat([x[x['name']=='skewness'] for x in REF],ignore_index=True)
    skew24 = skewdf[skewdf.duration=='24H']['value'].values
    skew1  = skewdf[skewdf.duration=='1H']['value'].values
    Xs = np.log(skew24/np.sqrt(var24))
    Ys = np.log(skew1/np.sqrt(var1))
    slope, intercept, rval, _, _ = linregress(Xs,Ys)
    xx = np.linspace(Xs.min(),Xs.max(),100)
    ax = axes[1,r]
    ax.scatter(Xs,Ys,s=10,alpha=0.65 )
    ax.plot(xx,intercept+slope*xx,color='red',lw=2)
    ax.text(0.03,0.97, f"ln(S$_{{1H}}$/σ$_{{1H}}$)" f" = {intercept:.3f}" f" + {slope:.3f}" f" ln(S$_{{24H}}$/σ$_{{24H}}$)\n\n" f"R$^2$ = {rval**2:.3f}", transform=ax.transAxes,fontsize=11,fontweight='bold',va='top',)

    if r==0:
        ax.set_ylabel("ln(Skew/SD 1H)",fontsize=14,fontweight='bold')

    ax.set_xlabel("ln(Skew/SD 24H)",fontsize=13,fontweight='bold')



    drydf = pd.concat([x[x['name'].isin(['probability_dry_0.1mm','probability_dry_1mm'])] for x in REF], ignore_index=True)
    p24 = drydf[drydf.duration=='24H']['value'].values
    p1  = drydf[drydf.duration=='1H']['value'].values
    Xd = np.log(p24/(1-p24))
    Yd = np.log(p1/(1-p1))
    slope, intercept, rval, _, _ = linregress(Xd,Yd)
    xx = np.linspace(Xd.min(),Xd.max(),100)
    ax = axes[2,r]
    ax.scatter(Xd,Yd,s=10,alpha=0.65)
    ax.plot(xx,intercept+slope*xx,color='red',lw=2)
    ax.text(0.03,0.97, f"logit(P$_{{1H}}$)" f" = {intercept:.3f}" f" + {slope:.3f}" f" logit(P$_{{24H}}$)\n\n" f"R$^2$ = {rval**2:.3f}",transform=ax.transAxes,fontsize=11,fontweight='bold',va='top')

    if r==0:
        ax.set_ylabel("logit(Pdry 1H)",fontsize=14,fontweight='bold')

    ax.set_xlabel("logit(Pdry 24H)",fontsize=13,fontweight='bold')


for ax in axes.ravel():
    ax.tick_params(labelsize=12)
    plt.setp(ax.get_xticklabels(),fontweight='bold')
    plt.setp(ax.get_yticklabels(),fontweight='bold')
    # ax.grid(alpha=.25)


# fig.text(0.03, 0.83, "Variance",rotation=90,fontsize=18,color='darkred',fontweight='bold',va='center')
# fig.text(0.03, 0.50, "Skewness", rotation=90, fontsize=18, color='navy', fontweight='bold', va='center')
# fig.text(0.03,0.18,"Dry Fraction",rotation=90,fontsize=18,color='darkgreen',fontweight='bold',va='center')
plt.tight_layout(rect=[0.05,0.02,1,1])
plt.savefig("/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/HOURLY_DAILY/PLOTS/Regional_Relationships_3x5.png", dpi=600,bbox_inches="tight")
plt.show()
