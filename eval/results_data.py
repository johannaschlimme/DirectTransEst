#%%
# RESULTS BY TASK

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('FinalResults.csv')

# Define unique colors for each Target Data value
colors = plt.cm.viridis(np.linspace(0, 1, len(df['Target Data'].unique())))
color_map = {target: colors[i] for i, target in enumerate(df['Target Data'].unique())}


# SOURCE ADNI
df_adni = df[df['Source Data'] == 'ADNI']

df_adni_ccfv = df_adni['Pre-FT CCFV']
df_adni_transrate = df_adni['Pre-FT TransRate']
df_adni_dice = df_adni['Pre-FT Dice']
df_adni_post = df_adni['Post-Fine-Tuning']

df_adni.loc[:, 'color'] = df_adni['Target Data'].apply(lambda x: color_map[x])

# correlation coefficients
adni_pears_ccfv = df_adni_ccfv.corr(df_adni_post, method='pearson')
adni_pears_transrate = df_adni_transrate.corr(df_adni_post, method='pearson')
adni_pears_dice = df_adni_dice.corr(df_adni_post, method='pearson')
adni_kendall_ccfv = df_adni_ccfv.corr(df_adni_post, method='kendall')
adni_kendall_transrate = df_adni_transrate.corr(df_adni_post, method='kendall')
adni_kendall_dice = df_adni_dice.corr(df_adni_post, method='kendall')

# SOURCE HCP
df_hcp = df[df['Source Data'] == 'HCP']

df_hcp_ccfv = df_hcp['Pre-FT CCFV']
df_hcp_transrate = df_hcp['Pre-FT TransRate']
df_hcp_dice = df_hcp['Pre-FT Dice']
df_hcp_post = df_hcp['Post-Fine-Tuning']

df_hcp.loc[:, 'color'] = df_hcp['Target Data'].apply(lambda x: color_map[x])

# correlation coefficients
hcp_pears_ccfv = df_hcp_ccfv.corr(df_hcp_post, method='pearson')
hcp_pears_transrate = df_hcp_transrate.corr(df_hcp_post, method='pearson')
hcp_pears_dice = df_hcp_dice.corr(df_hcp_post, method='pearson')
hcp_kendall_ccfv = df_hcp_ccfv.corr(df_hcp_post, method='kendall')
hcp_kendall_transrate = df_hcp_transrate.corr(df_hcp_post, method='kendall')
hcp_kendall_dice = df_hcp_dice.corr(df_hcp_post, method='kendall')

# SOURCE NACC
df_nacc = df[df['Source Data'] == 'NACC']

df_nacc_ccfv = df_nacc['Pre-FT CCFV']
df_nacc_transrate = df_nacc['Pre-FT TransRate']
df_nacc_dice = df_nacc['Pre-FT Dice']
df_nacc_post = df_nacc['Post-Fine-Tuning']

df_nacc.loc[:, 'color'] = df_nacc['Target Data'].apply(lambda x: color_map[x])

# correlation coefficients
nacc_pears_ccfv = df_nacc_ccfv.corr(df_nacc_post, method='pearson')
nacc_pears_transrate = df_nacc_transrate.corr(df_nacc_post, method='pearson')
nacc_pears_dice = df_nacc_dice.corr(df_nacc_post, method='pearson')
nacc_kendall_ccfv = df_nacc_ccfv.corr(df_nacc_post, method='kendall')
nacc_kendall_transrate = df_nacc_transrate.corr(df_nacc_post, method='kendall')
nacc_kendall_dice = df_nacc_dice.corr(df_nacc_post, method='kendall')

# SOURCE OASIS
df_oasis = df[df['Source Data'] == 'OASIS']

df_oasis_ccfv = df_oasis['Pre-FT CCFV']
df_oasis_transrate = df_oasis['Pre-FT TransRate']
df_oasis_dice = df_oasis['Pre-FT Dice']
df_oasis_post = df_oasis['Post-Fine-Tuning']

df_oasis.loc[:, 'color'] = df_oasis['Target Data'].apply(lambda x: color_map[x])

# correlation coefficients
oasis_pears_ccfv = df_oasis_ccfv.corr(df_oasis_post, method='pearson')
oasis_pears_transrate = df_oasis_transrate.corr(df_oasis_post, method='pearson')
oasis_pears_dice = df_oasis_dice.corr(df_oasis_post, method='pearson')
oasis_kendall_ccfv = df_oasis_ccfv.corr(df_oasis_post, method='kendall')
oasis_kendall_transrate = df_oasis_transrate.corr(df_oasis_post, method='kendall')
oasis_kendall_dice = df_oasis_dice.corr(df_oasis_post, method='kendall')


#%%
# Create scatter plots
fig, axs = plt.subplots(1, 6, figsize=(24, 4))
fig.text(1/6 * 1.5 + 0.018, 1, 'ADNI', ha='center', va='center', fontsize=14, fontweight='bold')
axs[0].scatter(df_adni_ccfv, df_adni_post, c=df_adni['color'])
axs[0].set_title('CC-FV')
axs[0].set_xlabel('Pearson: ' + str(round(adni_pears_ccfv, 3)) + '\nKendall: ' + str(round(adni_kendall_ccfv, 3)))
axs[0].set_ylabel('Post-Fine-Tuning DSC')
axs[1].scatter(df_adni_transrate, df_adni_post, c=df_adni['color'])
axs[1].set_title('TransRate')
axs[1].set_xlabel('Pearson: ' + str(round(adni_pears_transrate, 3)) + '\nKendall: ' + str(round(adni_kendall_transrate, 3)))
axs[2].scatter(df_adni_dice, df_adni_post, c=df_adni['color'])
axs[2].set_title('DirectTransEst')
axs[2].set_xlabel('Pearson: ' + str(round(adni_pears_dice, 3)) + '\nKendall: ' + str(round(adni_kendall_dice, 3)))

fig.text(1 - 1/6 * 1.5 + 0.012, 1, 'HCP', ha='center', va='center', fontsize=14, fontweight='bold')
axs[3].scatter(df_hcp_ccfv, df_hcp_post, c=df_hcp['color'])
axs[3].set_title('CC-FV')
axs[3].set_xlabel('Pearson: ' + str(round(hcp_pears_ccfv, 3)) + '\nKendall: ' + str(round(hcp_kendall_ccfv, 3)))
axs[4].scatter(df_hcp_transrate, df_hcp_post, c=df_hcp['color'])
axs[4].set_title('TransRate')
axs[4].set_xlabel('Pearson: ' + str(round(hcp_pears_transrate, 3)) + '\nKendall: ' + str(round(hcp_kendall_transrate, 3)))
axs[5].scatter(df_hcp_dice, df_hcp_post, c=df_hcp['color'])
axs[5].set_title('DirectTransEst')
axs[5].set_xlabel('Pearson: ' + str(round(hcp_pears_dice, 3)) + '\nKendall: ' + str(round(hcp_kendall_dice, 3)))

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 6, figsize=(24, 4))
fig.text(1/6 * 1.5 + 0.015, 1, 'NACC', ha='center', va='center', fontsize=14, fontweight='bold')
axs[0].scatter(df_nacc_ccfv, df_nacc_post, c=df_nacc['color'])
axs[0].set_title('CC-FV')
axs[0].set_xlabel('Pearson: ' + str(round(nacc_pears_ccfv, 3)) + '\nKendall: ' + str(round(nacc_kendall_ccfv, 3)))
axs[0].set_ylabel('Post-Fine-Tuning DSC')
axs[1].scatter(df_nacc_transrate, df_nacc_post, c=df_nacc['color'])
axs[1].set_title('TransRate')
axs[1].set_xlabel('Pearson: ' + str(round(nacc_pears_transrate, 3)) + '\nKendall: ' + str(round(nacc_kendall_transrate, 3)))
axs[2].scatter(df_nacc_dice, df_nacc_post, c=df_nacc['color'])
axs[2].set_title('DirectTransEst')
axs[2].set_xlabel('Pearson: ' + str(round(nacc_pears_dice, 3)) + '\nKendall: ' + str(round(nacc_kendall_dice, 3)))

fig.text(1 - 1/6 * 1.5 + 0.011, 1, 'OASIS', ha='center', va='center', fontsize=14, fontweight='bold')
axs[3].scatter(df_oasis_ccfv, df_oasis_post, c=df_oasis['color'])
axs[3].set_title('CC-FV')
axs[3].set_xlabel('Pearson: ' + str(round(oasis_pears_ccfv, 3)) + '\nKendall: ' + str(round(oasis_kendall_ccfv, 3)))
axs[4].scatter(df_oasis_transrate, df_oasis_post, c=df_oasis['color'])
axs[4].set_title('TransRate')
axs[4].set_xlabel('Pearson: ' + str(round(oasis_pears_transrate, 3)) + '\nKendall: ' + str(round(oasis_kendall_transrate, 3)))
axs[5].scatter(df_oasis_dice, df_oasis_post, c=df_oasis['color'])
axs[5].set_title('DirectTransEst')
axs[5].set_xlabel('Pearson: ' + str(round(oasis_pears_dice, 3)) + '\nKendall: ' + str(round(oasis_kendall_dice, 3)))

# Add a legend
sorted_color_map = dict(sorted(color_map.items()))
handles = [plt.Line2D([0], [0], marker='o', color='w', label=target, markersize=10, markerfacecolor=color) 
           for target, color in sorted_color_map.items()]
fig.legend(handles=handles, loc='lower center', ncol=len(color_map), bbox_to_anchor=(0.5, -0.1))

plt.tight_layout()
plt.show()


# %%
