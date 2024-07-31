import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


hr_data = pd.read_csv('./HR Data.csv')


hr_data.fillna(method='ffill', inplace=True)  


plt.figure(figsize=(10, 6))
sns.histplot(hr_data['Age'], bins=30, kde=True)
plt.title('Distribution of Employee Ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
corr_matrix = hr_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


hr_data['tenure_years'] = (pd.to_datetime('today') - pd.to_datetime(hr_data['start_date'])).dt.days / 365


mean_salary = hr_data['salary'].mean()
median_salary = hr_data['salary'].median()
print(f'Mean Salary: {mean_salary}, Median Salary: {median_salary}')

plt.figure(figsize=(10, 6))
hr_data['department'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Employee Count by Department')
plt.xlabel('Department')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = hr_data[['Age', 'salary', 'tenure_years']]
y = hr_data['left_company']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of Logistic Regression model: {accuracy}')
