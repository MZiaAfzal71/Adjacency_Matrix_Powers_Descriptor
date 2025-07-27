import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 9}
font1 = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 11}

font2 = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 13}

plt.rc('font', **font)
# plt.rcParams['xtick.major.size'] = 3
# plt.rcParams['xtick.major.width'] = 3
# plt.rcParams['ytick.major.size'] = 3
# plt.rcParams['ytick.major.width'] = 3
# plt.rcParams['xtick.minor.size'] = 10
# plt.rcParams['xtick.minor.width'] = 2

fig, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12))  = plt.subplots(2, 6, sharey=True, figsize=(15, 8))

# Load data
df = pd.read_excel("../../Output/New_Results.xlsx")

sp_random = '../../Split Indices/random_split_42.npz' # path containing random split indices
sp_scaffold = '../../Split Indices/scaffold_split_Murcko.npz' # path containing scaffold split indices

split_data = np.load(sp_random)
train_random, val_random, test_random = split_data['train_idx'], split_data['val_idx'], split_data['test_idx']

split_data = np.load(sp_scaffold)
train_scaffold, val_scaffold, test_scaffold = split_data['train_idx'], split_data['val_idx'], split_data['test_idx']

# combine val and test indices

tot_test_random = np.concatenate([val_random, test_random])
tot_test_scaffold = np.concatenate([val_scaffold, test_scaffold])

Train_df_random = df.loc[train_random, :]
Test_df_random = df.loc[tot_test_random, :]

Train_df_scaffold = df.loc[train_scaffold, :]
Test_df_scaffold = df.loc[tot_test_scaffold, :]

min_data = df['Observed'].min()
max_data = df['Observed'].max()

Train_Observed_random = Train_df_random['Observed']
Test_Observed_random = Test_df_random['Observed']

Train_Observed_scaffold = Train_df_scaffold['Observed']
Test_Observed_scaffold = Test_df_scaffold['Observed']


# ##################            SVR               ############################
# svr_maccs_train_random = Train_df_random['SVM_MACCS_random_Prediction']
# svr_maccs_test_random = Test_df_random['SVM_MACCS_random_Prediction']
#
# svr_morgan_train_random = Train_df_random['SVM_Morgan_random_Prediction']
# svr_morgan_test_random = Test_df_random['SVM_Morgan_random_Prediction']
#
# svr_mordred_train_random = Train_df_random['SVM_Mordred_random_Prediction']
# svr_mordred_test_random = Test_df_random['SVM_Mordred_random_Prediction']
#
# svr_coulomb_train_random = Train_df_random['SVM_Coulomb_random_Prediction']
# svr_coulomb_test_random = Test_df_random['SVM_Coulomb_random_Prediction']
#
# svr_wl_train_random = Train_df_random['SVM_WL_random_Prediction']
# svr_wl_test_random = Test_df_random['SVM_WL_random_Prediction']
#
# svr_our_train_random = Train_df_random['SVM_OurDescriptor_random_Prediction']
# svr_our_test_random = Test_df_random['SVM_OurDescriptor_random_Prediction']
#
#
# svr_maccs_train_scaffold = Train_df_scaffold['SVM_MACCS_scaffold_Prediction']
# svr_maccs_test_scaffold = Test_df_scaffold['SVM_MACCS_scaffold_Prediction']
#
# svr_morgan_train_scaffold = Train_df_random['SVM_Morgan_scaffold_Prediction']
# svr_morgan_test_scaffold = Test_df_random['SVM_Morgan_scaffold_Prediction']
#
# svr_mordred_train_scaffold = Train_df_random['SVM_Mordred_scaffold_Prediction']
# svr_mordred_test_scaffold = Test_df_random['SVM_Mordred_scaffold_Prediction']
#
# svr_coulomb_train_scaffold = Train_df_random['SVM_Coulomb_scaffold_Prediction']
# svr_coulomb_test_scaffold = Test_df_random['SVM_Coulomb_scaffold_Prediction']
#
# svr_wl_train_scaffold = Train_df_random['SVM_WL_scaffold_Prediction']
# svr_wl_test_scaffold = Test_df_random['SVM_WL_scaffold_Prediction']
#
# svr_our_train_scaffold = Train_df_random['SVM_OurDescriptor_scaffold_Prediction']
# svr_our_test_scaffold = Test_df_random['SVM_OurDescriptor_scaffold_Prediction']

##################            RF               ############################
rf_maccs_train_random = Train_df_random['RF_MACCS_random_Prediction']
rf_maccs_test_random = Test_df_random['RF_MACCS_random_Prediction']

rf_morgan_train_random = Train_df_random['RF_Morgan_random_Prediction']
rf_morgan_test_random = Test_df_random['RF_Morgan_random_Prediction']

rf_mordred_train_random = Train_df_random['RF_Mordred_random_Prediction']
rf_mordred_test_random = Test_df_random['RF_Mordred_random_Prediction']

rf_coulomb_train_random = Train_df_random['RF_Coulomb_random_Prediction']
rf_coulomb_test_random = Test_df_random['RF_Coulomb_random_Prediction']

rf_wl_train_random = Train_df_random['RF_WL_random_Prediction']
rf_wl_test_random = Test_df_random['RF_WL_random_Prediction']

rf_our_train_random = Train_df_random['RF_OurDescriptor_random_Prediction']
rf_our_test_random = Test_df_random['RF_OurDescriptor_random_Prediction']


rf_maccs_train_scaffold = Train_df_scaffold['RF_MACCS_scaffold_Prediction']
rf_maccs_test_scaffold = Test_df_scaffold['RF_MACCS_scaffold_Prediction']

rf_morgan_train_scaffold = Train_df_random['RF_Morgan_scaffold_Prediction']
rf_morgan_test_scaffold = Test_df_random['RF_Morgan_scaffold_Prediction']

rf_mordred_train_scaffold = Train_df_random['RF_Mordred_scaffold_Prediction']
rf_mordred_test_scaffold = Test_df_random['RF_Mordred_scaffold_Prediction']

rf_coulomb_train_scaffold = Train_df_random['RF_Coulomb_scaffold_Prediction']
rf_coulomb_test_scaffold = Test_df_random['RF_Coulomb_scaffold_Prediction']

rf_wl_train_scaffold = Train_df_random['RF_WL_scaffold_Prediction']
rf_wl_test_scaffold = Test_df_random['RF_WL_scaffold_Prediction']

rf_our_train_scaffold = Train_df_random['RF_OurDescriptor_scaffold_Prediction']
rf_our_test_scaffold = Test_df_random['RF_OurDescriptor_scaffold_Prediction']


# ##################            XGB               ############################
# xgb_maccs_train_random = Train_df_random['XGB_MACCS_random_Prediction']
# xgb_maccs_test_random = Test_df_random['XGB_MACCS_random_Prediction']
#
# xgb_morgan_train_random = Train_df_random['XGB_Morgan_random_Prediction']
# xgb_morgan_test_random = Test_df_random['XGB_Morgan_random_Prediction']
#
# xgb_mordred_train_random = Train_df_random['XGB_Mordred_random_Prediction']
# xgb_mordred_test_random = Test_df_random['XGB_Mordred_random_Prediction']
#
# xgb_coulomb_train_random = Train_df_random['XGB_Coulomb_random_Prediction']
# xgb_coulomb_test_random = Test_df_random['XGB_Coulomb_random_Prediction']
#
# xgb_wl_train_random = Train_df_random['XGB_WL_random_Prediction']
# xgb_wl_test_random = Test_df_random['XGB_WL_random_Prediction']
#
# xgb_our_train_random = Train_df_random['XGB_OurDescriptor_random_Prediction']
# xgb_our_test_random = Test_df_random['XGB_OurDescriptor_random_Prediction']
#
#
# xgb_maccs_train_scaffold = Train_df_scaffold['XGB_MACCS_scaffold_Prediction']
# xgb_maccs_test_scaffold = Test_df_scaffold['XGB_MACCS_scaffold_Prediction']
#
# xgb_morgan_train_scaffold = Train_df_random['XGB_Morgan_scaffold_Prediction']
# xgb_morgan_test_scaffold = Test_df_random['XGB_Morgan_scaffold_Prediction']
#
# xgb_mordred_train_scaffold = Train_df_random['XGB_Mordred_scaffold_Prediction']
# xgb_mordred_test_scaffold = Test_df_random['XGB_Mordred_scaffold_Prediction']
#
# xgb_coulomb_train_scaffold = Train_df_random['XGB_Coulomb_scaffold_Prediction']
# xgb_coulomb_test_scaffold = Test_df_random['XGB_Coulomb_scaffold_Prediction']
#
# xgb_wl_train_scaffold = Train_df_random['XGB_WL_scaffold_Prediction']
# xgb_wl_test_scaffold = Test_df_random['XGB_WL_scaffold_Prediction']
#
# xgb_our_train_scaffold = Train_df_random['XGB_OurDescriptor_scaffold_Prediction']
# xgb_our_test_scaffold = Test_df_random['XGB_OurDescriptor_scaffold_Prediction']

################     MACCS      #####################
# sns.scatterplot(x=Train_Observed_random, y=svr_maccs_train_random, alpha=1, ax=ax1)
# sns.scatterplot(x=Test_Observed_random, y=svr_maccs_test_random, alpha=0.4, ax=ax1, color='orange')
sns.scatterplot(x=Train_Observed_random, y=rf_maccs_train_random, alpha=1, ax=ax1)
sns.scatterplot(x=Test_Observed_random, y=rf_maccs_test_random, alpha=0.4, ax=ax1, color='orange')
# sns.scatterplot(x=Train_Observed_random, y=xgb_maccs_train_random, alpha=1, ax=ax1)
# sns.scatterplot(x=Test_Observed_random, y=xgb_maccs_test_random, alpha=0.4, ax=ax1, color='orange')
ax1.plot([min_data, max_data], [min_data, max_data], 'r--')
# sns.scatterplot(x=Train_Observed_scaffold, y=svr_maccs_train_scaffold, alpha=1, ax=ax7)
# sns.scatterplot(x=Test_Observed_scaffold, y=svr_maccs_test_scaffold, alpha=0.4, ax=ax7, color='orange')
sns.scatterplot(x=Train_Observed_scaffold, y=rf_maccs_train_scaffold, alpha=1, ax=ax7)
sns.scatterplot(x=Test_Observed_scaffold, y=rf_maccs_test_scaffold, alpha=0.4, ax=ax7, color='orange')
# sns.scatterplot(x=Train_Observed_scaffold, y=xgb_maccs_train_scaffold, alpha=1, ax=ax7)
# sns.scatterplot(x=Test_Observed_scaffold, y=xgb_maccs_test_scaffold, alpha=0.4, ax=ax7, color='orange')
ax7.plot([min_data, max_data], [min_data, max_data], 'r--')

# Hide the right and top spines
ax1.spines[['right', 'top']].set_visible(False)
# ax1.spines[['left', 'bottom']].set_linewidth(2)

ax7.spines[['right']].set_visible(False)
# ax7.spines[['left', 'bottom']].set_linewidth(2)
#
ax1.set_ylabel('Scaffold', fontdict=font1)
ax7.set_ylabel('Random', fontdict=font1)

ax1.set_title('MACCS', fontdict=font2)

ax7.set_xlabel('')

################      Morgan      #####################
# sns.scatterplot(x=Train_Observed_random, y=svr_morgan_train_random, alpha=1, ax=ax2)
# sns.scatterplot(x=Test_Observed_random, y=svr_morgan_test_random, alpha=0.4, ax=ax2, color='orange')
sns.scatterplot(x=Train_Observed_random, y=rf_morgan_train_random, alpha=1, ax=ax2)
sns.scatterplot(x=Test_Observed_random, y=rf_morgan_test_random, alpha=0.4, ax=ax2, color='orange')
# sns.scatterplot(x=Train_Observed_random, y=xgb_morgan_train_random, alpha=1, ax=ax2)
# sns.scatterplot(x=Test_Observed_random, y=xgb_morgan_test_random, alpha=0.4, ax=ax2, color='orange')
ax2.plot([min_data, max_data], [min_data, max_data], 'r--')
# sns.scatterplot(x=Train_Observed_scaffold, y=svr_maccs_train_scaffold, alpha=1, ax=ax8)
# sns.scatterplot(x=Test_Observed_scaffold, y=svr_maccs_test_scaffold, alpha=0.4, ax=ax8, color='orange')
sns.scatterplot(x=Train_Observed_scaffold, y=rf_maccs_train_scaffold, alpha=1, ax=ax8)
sns.scatterplot(x=Test_Observed_scaffold, y=rf_maccs_test_scaffold, alpha=0.4, ax=ax8, color='orange')
# sns.scatterplot(x=Train_Observed_scaffold, y=xgb_maccs_train_scaffold, alpha=1, ax=ax8)
# sns.scatterplot(x=Test_Observed_scaffold, y=xgb_maccs_test_scaffold, alpha=0.4, ax=ax8, color='orange')
ax8.plot([min_data, max_data], [min_data, max_data], 'r--')

# Hide the right and top spines
ax2.spines[['right', 'top']].set_visible(False)
# ax2.spines[['left', 'bottom']].set_linewidth(2)

ax8.spines[['right']].set_visible(False)
# ax8.spines[['left', 'bottom']].set_linewidth(2)
#

ax2.set_title('Morgan', fontdict=font2)

ax8.set_xlabel('')


#################      Mordred    #####################
# sns.scatterplot(x=Train_Observed_random, y=svr_mordred_train_random, alpha=1, ax=ax3)
# sns.scatterplot(x=Test_Observed_random, y=svr_mordred_test_random, alpha=0.4, ax=ax3, color='orange')
sns.scatterplot(x=Train_Observed_random, y=rf_mordred_train_random, alpha=1, ax=ax3)
sns.scatterplot(x=Test_Observed_random, y=rf_mordred_test_random, alpha=0.4, ax=ax3, color='orange')
# sns.scatterplot(x=Train_Observed_random, y=xgb_mordred_train_random, alpha=1, ax=ax3)
# sns.scatterplot(x=Test_Observed_random, y=xgb_mordred_test_random, alpha=0.4, ax=ax3, color='orange')
ax3.plot([min_data, max_data], [min_data, max_data], 'r--')
# sns.scatterplot(x=Train_Observed_scaffold, y=svr_mordred_train_scaffold, alpha=1, ax=ax9)
# sns.scatterplot(x=Test_Observed_scaffold, y=svr_mordred_test_scaffold, alpha=0.4, ax=ax9, color='orange')
sns.scatterplot(x=Train_Observed_scaffold, y=rf_mordred_train_scaffold, alpha=1, ax=ax9)
sns.scatterplot(x=Test_Observed_scaffold, y=rf_mordred_test_scaffold, alpha=0.4, ax=ax9, color='orange')
# sns.scatterplot(x=Train_Observed_scaffold, y=xgb_mordred_train_scaffold, alpha=1, ax=ax9)
# sns.scatterplot(x=Test_Observed_scaffold, y=xgb_mordred_test_scaffold, alpha=0.4, ax=ax9, color='orange')
ax9.plot([min_data, max_data], [min_data, max_data], 'r--')

# Hide the right and top spines
ax3.spines[['right', 'top']].set_visible(False)
# ax3.spines[['left', 'bottom']].set_linewidth(2)

ax9.spines[['right']].set_visible(False)
# ax9.spines[['left', 'bottom']].set_linewidth(2)
#

ax3.set_title('Mordred', fontdict=font2)

ax9.set_xlabel('')

###################     Coulomb     #####################
# sns.scatterplot(x=Train_Observed_random, y=svr_coulomb_train_random, alpha=1, ax=ax4)
# sns.scatterplot(x=Test_Observed_random, y=svr_coulomb_test_random, alpha=0.4, ax=ax4, color='orange')
sns.scatterplot(x=Train_Observed_random, y=rf_coulomb_train_random, alpha=1, ax=ax4)
sns.scatterplot(x=Test_Observed_random, y=rf_coulomb_test_random, alpha=0.4, ax=ax4, color='orange')
# sns.scatterplot(x=Train_Observed_random, y=xgb_coulomb_train_random, alpha=1, ax=ax4)
# sns.scatterplot(x=Test_Observed_random, y=xgb_coulomb_test_random, alpha=0.4, ax=ax4, color='orange')
ax4.plot([min_data, max_data], [min_data, max_data], 'r--')
# sns.scatterplot(x=Train_Observed_scaffold, y=svr_coulomb_train_scaffold, alpha=1, ax=ax10)
# sns.scatterplot(x=Test_Observed_scaffold, y=svr_coulomb_test_scaffold, alpha=0.4, ax=ax10, color='orange')
sns.scatterplot(x=Train_Observed_scaffold, y=rf_coulomb_train_scaffold, alpha=1, ax=ax10)
sns.scatterplot(x=Test_Observed_scaffold, y=rf_coulomb_test_scaffold, alpha=0.4, ax=ax10, color='orange')
# sns.scatterplot(x=Train_Observed_scaffold, y=xgb_coulomb_train_scaffold, alpha=1, ax=ax10)
# sns.scatterplot(x=Test_Observed_scaffold, y=xgb_coulomb_test_scaffold, alpha=0.4, ax=ax10, color='orange')
ax10.plot([min_data, max_data], [min_data, max_data], 'r--')

# Hide the right and top spines
ax4.spines[['right', 'top']].set_visible(False)
# ax4.spines[['left', 'bottom']].set_linewidth(2)

ax10.spines[['right']].set_visible(False)
# ax10.spines[['left', 'bottom']].set_linewidth(2)
#

ax4.set_title('Coulomb', fontdict=font2)

ax10.set_xlabel('')

###################     WL     #####################
# sns.scatterplot(x=Train_Observed_random, y=svr_wl_train_random, alpha=1, ax=ax5)
# sns.scatterplot(x=Test_Observed_random, y=svr_wl_test_random, alpha=0.4, ax=ax5, color='orange')
sns.scatterplot(x=Train_Observed_random, y=rf_wl_train_random, alpha=1, ax=ax5)
sns.scatterplot(x=Test_Observed_random, y=rf_wl_test_random, alpha=0.4, ax=ax5, color='orange')
# sns.scatterplot(x=Train_Observed_random, y=xgb_wl_train_random, alpha=1, ax=ax5)
# sns.scatterplot(x=Test_Observed_random, y=xgb_wl_test_random, alpha=0.4, ax=ax5, color='orange')
ax5.plot([min_data, max_data], [min_data, max_data], 'r--')
# sns.scatterplot(x=Train_Observed_scaffold, y=svr_wl_train_scaffold, alpha=1, ax=ax11)
# sns.scatterplot(x=Test_Observed_scaffold, y=svr_wl_test_scaffold, alpha=0.4, ax=ax11, color='orange')
sns.scatterplot(x=Train_Observed_scaffold, y=rf_wl_train_scaffold, alpha=1, ax=ax11)
sns.scatterplot(x=Test_Observed_scaffold, y=rf_wl_test_scaffold, alpha=0.4, ax=ax11, color='orange')
# sns.scatterplot(x=Train_Observed_scaffold, y=xgb_wl_train_scaffold, alpha=1, ax=ax11)
# sns.scatterplot(x=Test_Observed_scaffold, y=xgb_wl_test_scaffold, alpha=0.4, ax=ax11, color='orange')
ax11.plot([min_data, max_data], [min_data, max_data], 'r--')

# Hide the right and top spines
ax5.spines[['right', 'top']].set_visible(False)
# ax5.spines[['left', 'bottom']].set_linewidth(2)

ax11.spines[['right']].set_visible(False)
# ax11.spines[['left', 'bottom']].set_linewidth(2)
#

ax5.set_title('WL', fontdict=font2)

ax11.set_xlabel('')

##############     Our     #####################
# sns.scatterplot(x=Train_Observed_random, y=svr_our_train_random, alpha=1, ax=ax6)
# sns.scatterplot(x=Test_Observed_random, y=svr_our_test_random, alpha=0.4, ax=ax6, color='orange')
sns.scatterplot(x=Train_Observed_random, y=rf_our_train_random, alpha=1, ax=ax6)
sns.scatterplot(x=Test_Observed_random, y=rf_our_test_random, alpha=0.4, ax=ax6, color='orange')
# sns.scatterplot(x=Train_Observed_random, y=xgb_our_train_random, alpha=1, ax=ax6)
# sns.scatterplot(x=Test_Observed_random, y=xgb_our_test_random, alpha=0.4, ax=ax6, color='orange')
ax6.plot([min_data, max_data], [min_data, max_data], 'r--')
# sns.scatterplot(x=Train_Observed_scaffold, y=svr_our_train_scaffold, alpha=1, ax=ax12)
# sns.scatterplot(x=Test_Observed_scaffold, y=svr_our_test_scaffold, alpha=0.4, ax=ax12, color='orange')
sns.scatterplot(x=Train_Observed_scaffold, y=rf_our_train_scaffold, alpha=1, ax=ax12)
sns.scatterplot(x=Test_Observed_scaffold, y=rf_our_test_scaffold, alpha=0.4, ax=ax12, color='orange')
# sns.scatterplot(x=Train_Observed_scaffold, y=xgb_our_train_scaffold, alpha=1, ax=ax12)
# sns.scatterplot(x=Test_Observed_scaffold, y=xgb_our_test_scaffold, alpha=0.4, ax=ax12, color='orange')
ax12.plot([min_data, max_data], [min_data, max_data], 'r--')

# Hide the right and top spines
ax6.spines[['right', 'top']].set_visible(False)
# ax6.spines[['left', 'bottom']].set_linewidth(2)

ax12.spines[['right']].set_visible(False)
# ax12.spines[['left', 'bottom']].set_linewidth(2)
#

ax6.set_title('Our', fontdict=font2)

ax12.set_xlabel('')

#
fig.text(0.5, 0.04, r'Observed Boiling Points $^{\circ}$C', ha='center', fontdict=font2)
fig.text(0.05, 0.5, r'Predicted Boiling Points $^{\circ}$C', va='center', rotation='vertical', fontdict=font2)
#
plt.subplots_adjust(wspace=0.05, hspace=0)
plt.savefig("RF_Plots_new.png", bbox_inches="tight")
plt.show()
