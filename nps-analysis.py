#nps-analysis.py
#!/python

# Import module and intiante data frame
import scipy 
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency
from pylab import rcParams
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


df = pd.read_csv('../datasets/SAFS_Final/2017/Student Feedback Surveys-Superview.csv')
print(df)

#
# First conduct a brief analysis
#

print('Head of data:\n{0}\n\nTail of data:{1}'.format(df.head(), df.tail()))

#
# Acknowledge NA data
#

df = df.dropna()

df.info()

df.describe()

df.shape

#
# How many students atteneded in indiv tracks
#

track_total_df = pd.crosstab(df['Track'], df['Week'])
print('Total students in each track:\n {}'.format(track_total_df))

#
# Now what is the total sum of all students?
#

games_track = dict(pd.crosstab(df['Week'], df['Track']))['Games'].values
max(games_track) # 36

mobile_track = dict(pd.crosstab(df['Week'], df['Track']))['Apps'].values
max(mobile_track) # 183

vr_track = dict(pd.crosstab(df['Week'], df['Track']))['VR'].values
max(vr_track) # 10

e_games_track = dict(pd.crosstab(df['Week'], df['Track']))['Games, Explorer'].values
max(e_games_track) # 7

e_mobile_track = dict(pd.crosstab(df['Week'], df['Track']))['Apps, Explorer'].values
max(e_mobile_track) # 42


sum_of_students = max(games_track) + max(mobile_track) + max(vr_track) + max(e_games_track) + max(e_mobile_track) # 278

#
# Where did students come from
#

df.groupby(['Location']).count().plot(kind='bar')

#
# How many promoters are there than detractors in 2017 data
#

df = df.sort_values("Rating (Num)")

print('Head of data:\n{0}\n\nTail of data:{1}'.format(df.head(), df.tail()))

#
# Get rid of #Error, as string is conflictign the sum
#

df = df[~df['Rating (Num)'].str.contains("#Error")]

print('Head of data:\n{0}\n\nTail of data:{1}'.format(df.head(), df.tail()))

#
# Find Number of Promoters, Passives and Detractors
#

# Promoter (9 – 10)
# Passive (7 – 8)
# Detractor (1 – 6)

Promoters = df[pd.to_numeric(df['Rating (Num)']) >= 9]
Passive = df[(pd.to_numeric(df['Rating (Num)']) >= 7) & (pd.to_numeric(df['Rating (Num)']) <= 8)]
Detractors = df[pd.to_numeric(df['Rating (Num)']) < 7]

P_sub_D = len(Promoters) - len(Detractors)
NPS_Sum = len(Promoters) + len(Passive) + len(Detractors)

NPS = round(P_sub_D / NPS_Sum*100, 5)

print(f'NPS: {NPS}') # 44.52347

# The overall NPS score is good compared to average NPS scores


#
# Determine NPS score on a weekly basis
#

one_promoters = df[(pd.to_numeric(df['Rating (Num)']) >= 9) & (df['Week']=='Week 1')] # 122

one_passive = df[(pd.to_numeric(df['Rating (Num)']) >= 7) & (pd.to_numeric(df['Rating (Num)']) <= 8) & (df['Week']=='Week 1')] #128

one_detractors = df[(pd.to_numeric(df['Rating (Num)']) < 7) & (df['Week']=='Week 1')] # 122

# NPS = (Promoters - Detractors)  (Promoters + Passives + Detractors)
week_one_NPS = (len(one_promoters - one_detractors)) / (len(one_promoters) + len(one_passive) + len(one_detractors)) * 100

print(f'NPS score for week one: {str(week_one_NPS)}') # 53.62318840579711

#
# is there any positive corelation over time in the program, if so we can use pearsonr correlation? 
#

# Goodness-of-Fit Tests
# Curve Fitting
# Spearman's Rank Correlation
# Chi-square

sns.pairplot(df) # There is no linear relationship and the data is not normally distributed



rcParams['figure.figsize']=5, 4
sns.set_style('whitegrid')

print('Head of data:\n{0}\n\nTail of data:{1}'.format(df.head(), df.tail()))

location = df['Location']
track = df['Track']
week = df['Week']
rating = df['Rating (Num)']
schedule = df['Schedule Pacing']




#
# Chi-square test for independence
#

table = pd.crosstab(rating, schedule)

chi2, p, dof, expected = chi2_contingency(table.values) # Ch-square Statistic 188.526 p_value 0.000, Independent: False

table = pd.crosstab(week, rating)

chi2, p, dof, expected = chi2_contingency(table.values) # Ch-square Statistic 76.467 p_value 0.279, Independent: True

table = pd.crosstab(df['Week'], df['Schedule Pacing'])

chi2, p, dof, expected = chi2_contingency(table.values) # Ch-square Statistic 84.528 p_value 0.000, Independent: False



# We can use spearman Rank Correlation because we two ranked variables we want to find a corelation 


spearmanr_coefficient, p_value = spearmanr(week, rating) # -0.012741049530357226

spearmanr_coefficient, p_value = spearmanr(week, schedule) # -0.05858496761128574

spearmanr_coefficient, p_value = spearmanr(week, location) # 0.004045223031916859

spearmanr_coefficient, p_value = spearmanr(location, schedule) # -0.010961074073360139


#
#
#

sns.pairplot(df[['Location','Track','Week','Rating (Num)', 'Schedule Pacing']])

#
# Students from which location are more likely to give higher ratings?
#

ratloc = pd.crosstab(df['Location'], df['Rating (Num)'])

