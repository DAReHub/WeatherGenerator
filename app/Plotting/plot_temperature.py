import pandas as pd
import calendar
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from pathlib import Path


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
        obs_path: str,
        realisations: pd.DataFrame,
        EASTING: int,
        NORTHING: int,
        n_realizations: int,
        output_dir: Path
):
    """
    obs_path:      parquet of observed monthly statistics (columns: easting, northing, variable, mean)
    realisations:  daily average temperature, DatetimeIndex with columns R_1 ... R_n
    """
    PAST = pd.read_parquet(obs_path)
    obs_mean = PAST[
        (PAST['easting']==EASTING) & (PAST['northing']==NORTHING) & (PAST['variable']=='temp_avg')
    ].reset_index(drop=True)

    real_cols = [f"R_{i+1}" for i in range(n_realizations)]
    realisations = realisations.copy()
    realisations.index = pd.to_datetime(realisations.index)
    monthly_mean_real = realisations.groupby(realisations.index.month)[real_cols].mean()
    monthly_mean_real.index.name = 'Month'
    monthly_mean_real = monthly_mean_real.reset_index()

    month_names = [calendar.month_abbr[i] for i in range(1,13)]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    plt.rcParams['font.weight'] = 'bold'

    plot_stat(
        ax,
        monthly_mean_real['Month'],
        monthly_mean_real,
        obs_mean['mean'],
        'Monthly Mean Temperature',
        'Temperature (°C)',
        real_cols,
        month_names
    )
    # plt.show()
    plt.savefig(output_dir / f"E{EASTING}_N{NORTHING}_temperature_plot.png")