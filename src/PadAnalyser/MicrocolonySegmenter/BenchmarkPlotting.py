import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_theme(style="ticks")

sns.relplot(
    data=df,
    kind="scatter",
    x='threshold', y='Mean errors per cell',
    hue='label', 
    col='split_dist',
    row='sigma',
    palette="rocket",
)
plt.savefig(os.path.join(results_folder, f'{OPTIMIZATION_ROUND}_all_mean_error.png'), bbox_inches='tight', dpi=300)
plt.close()

sns.relplot(
    data=df,
    kind="scatter",
    x='threshold', y='iou',
    hue='label', 
    col='split_dist',
    row='sigma',
    palette="rocket",
)
plt.savefig(os.path.join(results_folder, f'{OPTIMIZATION_ROUND}_all_iou.png'), bbox_inches='tight', dpi=300)
plt.close()

sns.relplot(
    data=df,
    kind="scatter",
    x='threshold', y='matched_iou',
    hue='label', 
    col='split_dist',
    row='sigma',
    palette="rocket",
)
plt.savefig(os.path.join(results_folder, f'{OPTIMIZATION_ROUND}_all_matched_iou.png'), bbox_inches='tight', dpi=300)
plt.close()

sns.relplot(
    data=df,
    kind="scatter",
    x='threshold', y='Area error',
    hue='label', 
    col='split_dist',
    row='sigma',
    palette="rocket",
)
plt.savefig(os.path.join(results_folder, f'{OPTIMIZATION_ROUND}_all_area_error.png'), bbox_inches='tight', dpi=300)
plt.close()
