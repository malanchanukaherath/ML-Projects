
"""### Importing the necessary libraries"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Pandas = data handling library
#Used to work with tables (rows & columns) like Excel or CSV
#Provides DataFrame and Series
#Used for:
#Reading CSV / Excel files
#Cleaning data
#Filtering, grouping, aggregating

#NumPy = numerical computing library
#Used for arrays, matrices, and math operations
#Much faster than normal Python lists
#Used for:
#Mathematical calculations
#Statistical operations
#Supporting ML algorithms

#Matplotlib = basic plotting library
#Used to create graphs and charts
#Line plots, bar charts, histograms, scatter plots
#plt controls figure creation and display

#Seaborn = advanced visualization library
#Built on top of Matplotlib
#Makes better-looking, statistical graphs
#Works very well with Pandas DataFrames
#Used for:
#Heatmaps
#Boxplots
#Correlation plots

"""### Loading the necessary data"""

df = pd.read_csv('preprocessing.csv')

"""### Doing the basic inspection"""

#Run each of the following one by one and observe the outcome
df.head() #Shows the first 5 rows of the dataset
#df.tail() #Shows the last 5 rows of the dataset
#df.shape #Returns the size of the DataFrame (rows, columns)
#df.columns #Displays all column names
#df.info() #Total number of rows, Column names, Data type of each column (int, float, object, etc.), Number of non-null values, Memory usage

"""###Getting the summary statistics and unique values"""

#Run each of the following one-by-one and observe the outcome

#df.describe() #Gives statistical summary of numerical columns
#df.describe(include='object') #Gives summary of categorical (text) columns
#df.nunique() #Returns the number of unique values per column
#df['city'].value_counts() #Counts how many times each city appears
df['gender'].value_counts()
#df['age'].value_counts()

"""### Identifying missing values

"""

#df.isnull() #Creates a True / False map for missing values
#df.isnull().sum() #Total number of missing values per column
#f.isnull().mean()  #Calculates the average of missing values per column
df.isnull().mean() * 100  #Converts the proportion into a percentage

"""### Handling missing values"""

#df.dropna()  #Removes entire rows that contain any missing value.
#df1 = df.dropna()
#df1.info()
#df.info()

# Fill numeric with mean/median
#df['age'].fillna(df['age'].median(), inplace=True) #Finds missing values in age, Replaces them with the median age, Updates the original DataFrame
#df.info()
#df['cgpa'].fillna(df['cgpa'].mean(), inplace=True) #Finds missing values in cgpa, Replaces them with the median cgpa, Updates the original DataFrame
#df.info()

# Fill categorical with mode
#df['city'].fillna(df['city'].mode()[0], inplace=True)  #What's inplace = True here is Returns a new Series, df remains unchanged
df.info()

"""### Detecting outliers

"""

sns.boxplot(x=df['age']) #Creates a box plot for the age column, x= means the box is drawn horizontally
                         #Seaborn automatically: Calculates quartiles, Detects outliers
plt.show() #Displays the plot

"""###Using the IQR method"""

Q1 = df['age'].quantile(0.25) #The value below which 25% of the ages fall.
Q3 = df['age'].quantile(0.75) #The value below which 75% of the ages fall.
IQR = Q3 - Q1 #Measures the spread of the middle 50% of the data.

df1 = df[(df['age'] >= Q1 - 1.5*IQR) & (df['age'] <= Q3 + 1.5*IQR)] #Interpret this
                      #Keep only the rows where age lies within the normal range defined by the IQR method.
                      #Lower bound = Q1 - 1.5 × IQR
                      #Upper bound = Q3 + 1.5 × IQR
                      #Any age outside this range is considered an outlier and removed
                      #This is the standard statistical rule for detecting outliers.
df1.info()

"""###Lable encoding

"""

from sklearn.preprocessing import LabelEncoder #A tool from scikit-learn
                                               #Converts categorical text labels → numbers
                                               #Required because ML models only understand numbers

le = LabelEncoder()  #Creates an encoder object
                     #It will learn the mapping from text → numbers
df['gender'] = le.fit_transform(df['gender']) #Encode the gender column
                                              #fit() → finds unique categories in gender
                                              #transform() → converts them into integers
df.head()

"""### One-hot encoding"""

df = pd.get_dummies(df, columns=['city'], drop_first=True) #Applies One-Hot Encoding to the city column.
df.head()
#pd.get_dummies(...):-  Converts categorical text values → binary (0/1) columns
                       #Each unique city becomes its own column
#columns=['city']:- Specifies which column to encode
                   #Other columns remain unchanged
#drop_first=True:- Drops one dummy column, Prevents dummy variable trap (multicollinearity)
                  #The dropped city becomes the reference category
                  #If you had 3 cities → you get 2 new columns

"""###Feature scaling - Standardisation"""

from sklearn.preprocessing import StandardScaler #A tool from scikit-learn
                                                 #Used to standardize numerical features
                                                 #Transforms data to:
                                                 #Mean = 0
                                                 #Standard deviation = 1
                                                 #z=(x−μ​)/σ
                                                 #x → the original value
                                                 #(example: a student’s age = 22)
                                                 #μ → the mean (average) of all values
                                                 #(example: average age = 20)
                                                 #σ → the standard deviation
                                                 #(how spread out the values are)
                                                 #z → the standardized value (z-score)

scaler = StandardScaler()
df[['age']] = scaler.fit_transform(df[['age']])  #fit() → calculates mean & std of age
                                                 #transform() → applies standardization
                                                 #[['age']] keeps it 2D, which sklearn requires
                                                 #Scaled values replace original age values
df.head()

"""###Feature scaling - min-max normalisation"""

from sklearn.preprocessing import MinMaxScaler #Rescales values to a fixed range, usually 0 to 1
                                               #x′=(x-xmin​)/(xma​x−xmin)

scaler = MinMaxScaler()  #Initializes the scaler
df[['cgpa']] = scaler.fit_transform(df[['cgpa']]) #fit() → finds min & max CGPA
                                                  #transform() → rescales values between 0 and 1
                                                  #Original cgpa values are replaced
df.head()

"""###Detecting multi-colinearity"""

corr = df.corr(numeric_only=True) #Calculates pairwise correlation (Pearson by default)
                                  #Uses only numeric columns
                                  #Safely ignores:
                                  #Categorical text columns
                                  #Object-type columns
#What is correlation?
#Measures linear relationship between two variables
#Range: -1 to +1
#  Value	 Meaning
#    +1	   Strong positive
#     0	   No relationship
#    -1	   Strong negative

#corr = df.corr()                  #What will happen if this runs?
sns.heatmap(corr, cmap='coolwarm') #Visualizes correlation matrix
                                   #Colors represent strength:
                                   #🔴 red → positive correlation
                                   #🔵 blue → negative correlation
                                   #coolwarm makes contrasts clear
plt.show()

"""###Drop highly correlated features"""

print(df.columns)

corr = df.corr(numeric_only=True) #It calculates pairwise Pearson correlation coefficients between all numeric columns in DataFrame and prints them as a table.
print(corr)

df.drop(columns=['highly_correlated_feature'], inplace=True)  #Replace the column name as appropriate

#Absolute correlation value	Meaning
#≥ 0.8	Highly correlated ❌
#0.5 – 0.8	Moderately correlated ⚠️
#< 0.5	Weak / acceptable ✅

# No highly correlated features found, so no columns dropped

"""##Exercise:
###1.How would you remove a class imbalance if available?
###2. How would you introduce a new feature?
"""

from imblearn.over_sampling import SMOTE

X = df.drop('placed', axis=1)
y = df['placed']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(y_resampled.value_counts())

"""Resampling (Most common & exam-friendly)
🔹 (a) Oversampling (SMOTE)

Creates synthetic samples for the minority class.

✔ Keeps all data
✔ Avoids duplication
✔ Very popular in exams & industry
"""

#Example: Age group from age
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 20, 30, 40, 100],
    labels=['Teen', 'Young', 'Adult', 'Senior']
)

"""Create feature from existing columns

✔ Adds meaningful information
✔ Improves model understanding

###Saving the clean dataset
"""

df.to_csv("cleaned_data.csv", index=False)