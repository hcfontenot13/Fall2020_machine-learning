# Lab 7: Seaborn plotting tutorial

import seaborn as sns

sns.set_theme(style='darkgrid', font_scale=3)  # older version of sns: sns.set()
tips = sns.load_dataset('tips')

# Distribution plots
sns.displot(tips, x='total_bill', col='sex', kind='kde')
sns.displot(tips, x='total_bill', kind='kde')
sns.displot(tips, x='total_bill', kind='kde', cut=0)
sns.displot(tips, x='total_bill', stat='density')
sns.displot(tips, x='total_bill', y='size', kind='kde')
sns.displot(tips, x='total_bill', col='sex', kind='kde')

# Relational plots
sns.relplot(x='total_bill', y='tip', data=tips)
sns.relplot(x='total_bill', y='tip', hue='smoker', data=tips)
sns.relplot(x='total_bill', y='tip', hue='smoker', style='sex', data=tips, s=100)
sns.relplot(x='total_bill', y='tip', size='size', sizes=(15, 200), data=tips)

# Categorical plots
sns.catplot(x='day', y='total_bill', data=tips)
sns.catplot(x='day', y='total_bill', kind='swarm', data=tips)
sns.catplot(x='day', y='total_bill', hue='smoker', kind='swarm', data=tips)
sns.catplot(x='total_bill', y='day', hue='time', kind='swarm', data=tips)

sns.catplot(x='day',  y='total_bill', hue='smoker', kind='box', data=tips)
sns.catplot(x='day',  y='total_bill', hue='smoker', kind='violin', data=tips)
sns.catplot(x='day',  y='total_bill', hue='smoker', kind='violin', split=True, data=tips)

sns.catplot(x='day', y='total_bill', hue='smoker', col='time', kind='swarm', data=tips, s=7)
