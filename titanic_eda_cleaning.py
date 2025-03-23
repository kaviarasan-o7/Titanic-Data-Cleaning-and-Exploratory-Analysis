import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Combine train and test for easier feature engineering
combined_df = pd.concat([train_df, test_df], sort=False)

# --- Data Cleaning ---

# Fix the inplace warnings by using assignment instead
# Handle missing Age values
combined_df['Age'] = combined_df['Age'].fillna(combined_df['Age'].median())

# Handle missing Embarked values
combined_df['Embarked'] = combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0])

# Handle missing Fare values in the test set
combined_df['Fare'] = combined_df['Fare'].fillna(combined_df['Fare'].median())

# Convert Sex to numerical
combined_df['Sex'] = combined_df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode Embarked
combined_df = pd.get_dummies(combined_df, columns=['Embarked'], drop_first=True)

# Feature Engineering
combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1

# Extract Title from Name - Fix the escape sequence by using raw string r'...'
combined_df['Title'] = combined_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
combined_df['Title'] = combined_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combined_df['Title'] = combined_df['Title'].replace('Mlle', 'Miss')
combined_df['Title'] = combined_df['Title'].replace('Ms', 'Miss')
combined_df['Title'] = combined_df['Title'].replace('Mme', 'Mrs')
combined_df['Title'] = combined_df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
combined_df['Title'] = combined_df['Title'].fillna(0)  # Fill NaN if any

# Drop unnecessary columns
combined_df = combined_df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)

# Split back into train and test
train_cleaned = combined_df[:len(train_df)]
test_cleaned = combined_df[len(train_df):]

# --- Exploratory Data Analysis (EDA) ---

# Create a PDF to save all plots
from matplotlib.backends.backend_pdf import PdfPages

# Option 1: Save all plots to a single PDF file
# --- Streamlined EDA with fewer outputs ---
with PdfPages('titanic_eda_results.pdf') as pdf:
    
    # 1. Key Survival Factors - Figure 1
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Key Factors Affecting Survival on the Titanic', fontsize=16)
    
    # Survival rate by Sex
    sns.barplot(x='Sex', y='Survived', data=train_cleaned, ax=axes[0, 0])
    axes[0, 0].set_title('Survival Rate by Sex')
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(['Male', 'Female'])
    
    # Survival rate by Pclass
    sns.barplot(x='Pclass', y='Survived', data=train_cleaned, ax=axes[0, 1])
    axes[0, 1].set_title('Survival Rate by Passenger Class')
    
    # Age distribution by survival
    sns.violinplot(x='Survived', y='Age', data=train_cleaned, ax=axes[1, 0])
    axes[1, 0].set_title('Age Distribution by Survival')
    
    # Survival rate by Sex and Pclass (multivariate)
    sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train_cleaned, ax=axes[1, 1])
    axes[1, 1].set_title('Survival Rate by Sex and Class')
    axes[1, 1].legend(labels=['Male', 'Female'])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig()
    plt.close()
    
    # 2. Correlation and Relationships - Figure 2
    fig = plt.figure(figsize=(12, 10))
    
    # Correlation heatmap
    corr_matrix = train_cleaned.corr()
    mask = np.triu(corr_matrix)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask)
    plt.title('Correlation Between Features')
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # 3. Multivariate Analysis - Figure 4
    fig = plt.figure(figsize=(12, 10))
    
    # Survival rate by Sex and Pclass
    ax = fig.add_subplot(111)
    sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train_cleaned, ax=ax)
    ax.set_title('Survival Rate by Sex and Pclass')
    ax.set_xlabel('Passenger Class')
    ax.set_ylabel('Survival Rate')
    ax.legend(labels=['Male', 'Female'])
    
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # 5. Pair plot for numerical features - Figure 5
    numerical_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    pair_plot = sns.pairplot(train_cleaned[numerical_features])
    pdf.savefig(pair_plot.fig)
    plt.close()

print("EDA completed! All visualizations have been saved to 'titanic_eda_results.pdf'")

# Option 2: Display a summary of key findings
print("\nKey Findings from Titanic Dataset Analysis:")
print("-" * 50)

# Survival rate by sex
survival_by_sex = train_cleaned.groupby('Sex')['Survived'].mean()
print(f"Survival rate for females: {survival_by_sex[1]:.2%}")
print(f"Survival rate for males: {survival_by_sex[0]:.2%}")

# Survival rate by class
survival_by_class = train_cleaned.groupby('Pclass')['Survived'].mean()
print(f"\nSurvival rate by passenger class:")
for pclass, rate in survival_by_class.items():
    print(f"  Class {pclass}: {rate:.2%}")

# Age statistics
print(f"\nAge statistics:")
print(f"  Average age: {train_cleaned['Age'].mean():.1f} years")
print(f"  Youngest passenger: {train_cleaned['Age'].min():.1f} years")
print(f"  Oldest passenger: {train_cleaned['Age'].max():.1f} years")

# Correlation with survival
survival_corr = train_cleaned.corr()['Survived'].sort_values(ascending=False)
print("\nTop correlations with survival:")
for feature, corr in survival_corr.items():
    if feature != 'Survived':
        print(f"  {feature}: {corr:.3f}")