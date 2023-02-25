from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('UNSW-NB15_1.csv', header=None).dropna()

# Rename the columns
df.columns = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes',
              'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload',
              'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
              'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt',
              'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
              'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
              'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label']

# Drop the features that are not needed for analysis
df.drop(['srcip', 'sport', 'dstip', 'dsport', 'proto',
        'state', 'service'], axis=1, inplace=True)

# Check for null values
print(df.isnull().sum())

# Check the number of samples in each class
print(df['attack_cat'].value_counts())

# Visualize the number of samples in each class
plt.figure(figsize=(12, 6))
sns.countplot(x='attack_cat', data=df)
plt.xticks(rotation=90)
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(20, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='Blues')
plt.title('Correlation Matrix')
plt.show()

# Visualize the relationship between the variables and the target
plt.figure(figsize=(12, 6))
sns.boxplot(x='attack_cat', y='dur', data=df)
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='attack_cat', y='Sload', data=df)
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='attack_cat', y='Dload', data=df)
plt.xticks(rotation=90)
plt.show()

# Split the dataset into training and testing sets

x = df.drop('attack_cat', axis=1)
y = df['attack_cat']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Train a logistic regression model
print(df.isna().sum())
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pried = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pried)
print('Accuracy of logistic regression model:', accuracy)

# Train a random forest model

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pried = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pried)
print('Accuracy RandomForestClassifier:', accuracy)

# Train a decision tree model

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pried = dtc.predict(X_test)
accuracy = accuracy_score(y_test, y_pried)
print('Accuracy DecisionTreeClassifier:', accuracy)

# Train a support vector machine model

svm = SVC()
svm.fit(X_train, y_train)

y_pried = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pried)
print('Accuracy support vector machine model:', accuracy)
