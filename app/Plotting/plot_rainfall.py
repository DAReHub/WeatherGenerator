import pandas as pd
import numpy as np
import calendar
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gumbel_r
from pathlib import Path

def fit_gumbel(rvs):
    loc, scale = gumbel_r.fit(rvs)
    return loc, scale

def gumbel_return_level(mu, sigma, T):
    return mu - sigma * np.log(-np.log(1 - 1/T))

def plot_stat(ax, x, real_df, obs, title, ylabel, real_cols, month_names):
    real_lines = ax.plot(x,real_df[real_cols],color='darkgrey',linewidth=0.8,alpha=0.5)
    obs_line = ax.plot(x,obs,marker='o',color='red',linewidth=2,markersize=5,label='Observed')
    ax.set_title(title,fontsize=16,fontweight='bold')
    ax.set_xlabel('Month',fontsize=13,fontweight='bold')
    ax.set_ylabel(ylabel,fontsize=13,fontweight='bold')
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(month_names,fontsize=11,fontweight='bold')
    ax.tick_params(axis='y',labelsize=11)
    plt.setp(ax.get_yticklabels(),fontweight='bold')
    ax.legend(handles=[real_lines[0], obs_line[0]],labels=['Future', 'Past (2001 - 2020)'],fontsize=14,frameon=False)

def main(
        mean_rf_path: str,
        rprf_path: str,
        realisations: str,
        EASTING: int,
        NORTHING: int,
        n_realizations: int,
        return_periods: list,
        output_dir: Path
):
    MEAN_RF = pd.read_parquet(mean_rf_path)
    RPRF = pd.read_parquet(rprf_path)

    obs_mean = MEAN_RF[
        (MEAN_RF['easting']==EASTING) & (MEAN_RF['northing']==NORTHING)
    ].reset_index(drop=True)
    RPVAL_DF_OBS = RPRF[
        (RPRF['easting']==EASTING) & (RPRF['northing']==NORTHING)
    ].reset_index(drop=True)

    real_cols = [f"Realization_{i+1}" for i in range(n_realizations)]
    monthly_sum_real = realisations.groupby(['Year', 'Month'])[real_cols].sum().reset_index()
    monthly_mean_real = monthly_sum_real.groupby('Month')[real_cols].mean().reset_index()

    daily_real = (realisations.set_index('DateTime')[real_cols].resample('D').sum().reset_index())
    daily_real['Year'] = daily_real['DateTime'].dt.year
    REAL_MAX = []
    for col in real_cols:
        annual_max = daily_real.groupby('Year')[col].max()
        REAL_MAX.append(annual_max)

    GUMBEL_PAR_REAL = [fit_gumbel(REAL_MAX[i]) for i in range(len(REAL_MAX))]

    RPVAL_DF_REAL = []
    for b in range(len(GUMBEL_PAR_REAL)):
        rpval = [
            gumbel_return_level(
                mu=GUMBEL_PAR_REAL[b][0],
                sigma=GUMBEL_PAR_REAL[b][1],
                T=rp
            ) for rp in return_periods
        ]
        rpval_df = pd.DataFrame({'RF': rpval})
        RPVAL_DF_REAL.append(rpval_df)

    REAL_RP_DF = pd.concat(RPVAL_DF_REAL, axis=1)
    REAL_RP_DF.columns = real_cols
    REAL_RP_DF['RP'] = return_periods
    REAL_RP_DF_CLEAN = REAL_RP_DF.copy()
    df_long = REAL_RP_DF_CLEAN.melt(
        id_vars="RP",
        value_vars=real_cols,
        var_name="Realization",
        value_name="Value"
    )
    REAL_RP_MEAN = REAL_RP_DF_CLEAN[real_cols].mean(axis=1)

    RPVAL_DF_OBS = RPVAL_DF_OBS.drop(['easting','northing'],axis=1)
    month_names = [calendar.month_abbr[i] for i in range(1,13)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8),constrained_layout=True)
    axes = axes.flatten()
    plt.rcParams['font.weight'] = 'bold'

    plot_stat(
        axes[0],
        monthly_mean_real['Month'],
        monthly_mean_real,
        obs_mean['value'],
        'Monthly Mean Rainfall Depth',
        'Rainfall',
        real_cols,
        month_names
    )
    sns.boxplot(
        data=df_long,
        x='RP',
        y='Value',
        ax=axes[1],
        boxprops=dict(facecolor='none', edgecolor='blue'),
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        showfliers=False,
        flierprops=dict(markeredgecolor='black'),
        patch_artist=True
    )

    x_obs = np.arange(len(RPVAL_DF_OBS))
    x_future = np.arange(len(REAL_RP_MEAN))

    obs_line, = axes[1].plot(
        x_obs,
        RPVAL_DF_OBS['RF'],
        '-o',
        color='red',
        linewidth=1.8,
        markersize=8,
        label='Observed (2001 - 2020)'
    )
    future_line, = axes[1].plot(
        x_future,
        REAL_RP_MEAN,
        '-o',
        color='blue',
        linewidth=1.8,
        markersize=7,
        label='Future'
    )

    axes[1].set_title('24-Hour Rainfall Depths', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Return Period (Years)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Rainfall Depth (mm)', fontsize=13, fontweight='bold')
    axes[1].tick_params(labelsize=11)
    plt.setp(axes[1].get_yticklabels(),fontweight='bold')
    plt.setp(axes[1].get_xticklabels(),fontweight='bold')
    axes[1].legend(
        handles=[obs_line, future_line],
        labels=['Past (2001 - 2020)', 'Future'],
        fontsize=15,
        frameon=False,
        loc='upper left'
    )
    # plt.show()
    plt.savefig(output_dir / f"E{EASTING}_N{NORTHING}_rainfall_plot.png")