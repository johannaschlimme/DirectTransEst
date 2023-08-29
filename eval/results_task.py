
#%%
# RESULTS BY TASK

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('FinalResults.csv')

#%%
# CEREBRAL VENTRICULAR SYSTEM SEGMENTATION
df_cvs = df[df['Task'] == 'Ventricular System']

df_cvs_ccfv = df_cvs['Pre-FT CCFV']
df_cvs_transrate = df_cvs['Pre-FT TransRate']
df_cvs_dice = df_cvs['Pre-FT Dice']
df_cvs_post = df_cvs['Post-Fine-Tuning']

# correlation coefficients
cvs_pears_ccfv = df_cvs_ccfv.corr(df_cvs_post, method='pearson')
cvs_pears_transrate = df_cvs_transrate.corr(df_cvs_post, method='pearson')
cvs_pears_dice = df_cvs_dice.corr(df_cvs_post, method='pearson')
cvs_kendall_ccfv = df_cvs_ccfv.corr(df_cvs_post, method='kendall')
cvs_kendall_transrate = df_cvs_transrate.corr(df_cvs_post, method='kendall')
cvs_kendall_dice = df_cvs_dice.corr(df_cvs_post, method='kendall')

# scatter plot
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
fig.text(0.51, 0.99, 'Cerebral Ventricular System (CVS) Segmentation', ha='center', va='center', fontsize=12, fontweight='bold')

sns.regplot(x=df_cvs_ccfv, y=df_cvs_post, ax=axs[0])
axs[0].scatter(df_cvs_ccfv, df_cvs_post)
axs[0].set_title('CC-FV')
axs[0].set_xlabel('Pearson: ' + str(round(cvs_pears_ccfv, 3)) + '\nKendall: ' + str(round(cvs_kendall_ccfv, 3)))
axs[0].set_ylabel('Post-Fine-Tuning DSC')

sns.regplot(x=df_cvs_transrate, y=df_cvs_post, ax=axs[1])
axs[1].scatter(df_cvs_transrate, df_cvs_post)
axs[1].set_title('TransRate')
axs[1].set_xlabel('Pearson: ' + str(round(cvs_pears_transrate, 3)) + '\nKendall: ' + str(round(cvs_kendall_transrate, 3)))
axs[1].set_ylabel('')

sns.regplot(x=df_cvs_dice, y=df_cvs_post, ax=axs[2])
axs[2].scatter(df_cvs_dice, df_cvs_post)
axs[2].set_title('DirectTransEst')
axs[2].set_xlabel('Pearson: ' + str(round(cvs_pears_dice, 3)) + '\nKendall: ' + str(round(cvs_kendall_dice, 3)))
axs[2].set_ylabel('')
    
plt.show()


#%%
# HIPPOCAMPUS SEGMENTATION
df_hc = df[df['Task'] == 'Hippocampus']

df_hc_ccfv = df_hc['Pre-FT CCFV']
df_hc_transrate = df_hc['Pre-FT TransRate']
df_hc_dice = df_hc['Pre-FT Dice']
df_hc_post = df_hc['Post-Fine-Tuning']

# correlation coefficients
hc_pears_ccfv = df_hc_ccfv.corr(df_hc_post, method='pearson')
hc_pears_transrate = df_hc_transrate.corr(df_hc_post, method='pearson')
hc_pears_dice = df_hc_dice.corr(df_hc_post, method='pearson')
hc_kendall_ccfv = df_hc_ccfv.corr(df_hc_post, method='kendall')
hc_kendall_transrate = df_hc_transrate.corr(df_hc_post, method='kendall')
hc_kendall_dice = df_hc_dice.corr(df_hc_post, method='kendall')

# scatter plot
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
fig.text(0.51, 0.99, 'Hippocampus (HC) Segmentation', ha='center', va='center', fontsize=12, fontweight='bold')

sns.regplot(x=df_hc_ccfv, y=df_hc_post, ax=axs[0])
axs[0].scatter(df_hc_ccfv, df_hc_post)
axs[0].set_title('CC-FV')
axs[0].set_xlabel('Pearson: ' + str(round(hc_pears_ccfv, 3)) + '\nKendall: ' + str(round(hc_kendall_ccfv, 3)))
axs[0].set_ylabel('Post-Fine-Tuning DSC')

sns.regplot(x=df_hc_transrate, y=df_hc_post, ax=axs[1])
axs[1].scatter(df_hc_transrate, df_hc_post)
axs[1].set_title('TransRate')
axs[1].set_xlabel('Pearson: ' + str(round(hc_pears_transrate, 3)) + '\nKendall: ' + str(round(hc_kendall_transrate, 3)))
axs[1].set_ylabel('')

sns.regplot(x=df_hc_dice, y=df_hc_post, ax=axs[2])
axs[2].scatter(df_hc_dice, df_hc_post)
axs[2].set_title('DirectTransEst')
axs[2].set_xlabel('Pearson: ' + str(round(hc_pears_dice, 3)) + '\nKendall: ' + str(round(hc_kendall_dice, 3)))
axs[2].set_ylabel('')

plt.show()


#%%
# WHITE MATTER SEGMENTATION
df_wm = df[df['Task'] == 'White Matter']

df_wm_ccfv = df_wm['Pre-FT CCFV']
df_wm_transrate = df_wm['Pre-FT TransRate']
df_wm_dice = df_wm['Pre-FT Dice']
df_wm_post = df_wm['Post-Fine-Tuning']

# correlation coefficients
wm_pears_ccfv = df_wm_ccfv.corr(df_wm_post, method='pearson')
wm_pears_transrate = df_wm_transrate.corr(df_wm_post, method='pearson')
wm_pears_dice = df_wm_dice.corr(df_wm_post, method='pearson')
wm_kendall_ccfv = df_wm_ccfv.corr(df_wm_post, method='kendall')
wm_kendall_transrate = df_wm_transrate.corr(df_wm_post, method='kendall')
wm_kendall_dice = df_wm_dice.corr(df_wm_post, method='kendall')

# scatter plot
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
fig.text(0.51, 0.99, 'White Matter (WM) Segmentation', ha='center', va='center', fontsize=12, fontweight='bold')

sns.regplot(x=df_wm_ccfv, y=df_wm_post, ax=axs[0])
axs[0].scatter(df_wm_ccfv, df_wm_post)
axs[0].set_title('CC-FV')
axs[0].set_xlabel('Pearson: ' + str(round(wm_pears_ccfv, 3)) + '\nKendall: ' + str(round(wm_kendall_ccfv, 3)))
axs[0].set_ylabel('Post-Fine-Tuning DSC')

sns.regplot(x=df_wm_transrate, y=df_wm_post, ax=axs[1])
axs[1].scatter(df_wm_transrate, df_wm_post)
axs[1].set_title('TransRate')
axs[1].set_xlabel('Pearson: ' + str(round(wm_pears_transrate, 3)) + '\nKendall: ' + str(round(wm_kendall_transrate, 3)))
axs[1].set_ylabel('')

sns.regplot(x=df_wm_dice, y=df_wm_post, ax=axs[2])
axs[2].scatter(df_wm_dice, df_wm_post)
axs[2].set_title('DirectTransEst')
axs[2].set_xlabel('Pearson: ' + str(round(wm_pears_dice, 3)) + '\nKendall: ' + str(round(wm_kendall_dice, 3)))
axs[2].set_ylabel('')

plt.show()

# %%
