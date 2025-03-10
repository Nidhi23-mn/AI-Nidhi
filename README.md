
# Iris Flower Classification Using Random Forest

This project demonstrates how to classify Iris flowers into species based on their measurements using the Random Forest Classifier algorithm. The dataset consists of flower attributes such as sepal length, sepal width, petal length, and petal width. The goal is to predict the flower's species.

## Overview

We use the Iris dataset to train a machine learning model and evaluate its performance in predicting flower species. The Random Forest Classifier is used for this task. The steps include loading the data, training the model, evaluating accuracy, visualizing feature importance, and predicting species for new data.

## Steps in the Code

### 1. Importing Libraries

We begin by importing the necessary Python libraries:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
```

- **Pandas**: Used for data manipulation and reading the dataset.
- **Scikit-learn**: Provides tools for machine learning tasks, including splitting the data, training the model, and evaluating its accuracy.
- **Matplotlib/Seaborn**: Used for data visualization.

### 2. Load the Dataset

The dataset is loaded from a CSV file named `data.csv`. The dataset contains the following columns:

- `sepal length (cm)`
- `sepal width (cm)`
- `petal length (cm)`
- `petal width (cm)`
- `species` (target variable)

```python
# Load the Iris dataset
data = pd.read_csv('data.csv')

# Display the first few rows of the dataset
print(data.head())
```

### 3. Prepare Features and Target Variables

We separate the dataset into features (X) and target (y):

- **Features (X)**: The flower's attributes like sepal length, sepal width, petal length, and petal width.
- **Target (y)**: The species of the flowers (this is what we are trying to predict).

```python
# Features (sepal length, sepal width, petal length, petal width)
X = data.drop('species', axis=1)

# Target (species)
y = data['species']
```

### 4. Split the Data into Training and Testing Sets

We split the data into training and testing sets using an 80-20 ratio, meaning 80% of the data is used to train the model, and 20% is used to evaluate its performance.

```python
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. Train the Random Forest Classifier

We use the Random Forest Classifier to train the model on the training data. This classifier creates multiple decision trees and aggregates their results to make predictions.

```python
# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)
```

### 6. Make Predictions

After training, we use the model to make predictions on the test data. This helps assess how well the model generalizes to new, unseen data.

```python
# Make predictions on the test data
y_pred = model.predict(X_test)
```

### 7. Evaluate the Model's Accuracy

We calculate the accuracy score of the model by comparing the predicted species with the actual species in the test set. The accuracy is printed as a percentage to evaluate how well the classifier performs.

```python
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

### 8. Feature Importance

We visualize the importance of each feature (sepal length, sepal width, petal length, petal width) in the model’s decision-making process. This helps us understand which features contribute the most to the predictions.

```python
# Visualize feature importance
feature_importance = model.feature_importances_
features = X.columns

# Plot the feature importance
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance, y=features)
plt.title('Feature Importance')
plt.show()
```

### 9. Predict on a New Sample

Finally, we can use the trained model to predict the species of a new flower sample. This sample consists of the flower's measurements, and the model will output the predicted species.

```python
# Example new data (adjust the values based on your features)
sample = [[5.1, 3.5, 1.4, 0.2]]

# Predict the species for the new sample
predicted_species = model.predict(sample)
print(f'Predicted species for the new sample: {predicted_species[0]}')
```

### Example Output

If you run the code with the sample data `[[5.1, 3.5, 1.4, 0.2]]`, the model will predict the species and output something like:

```python
Predicted species for the new sample: setosa
```

## Requirements

To run this code, you need the following Python libraries:

- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

You can install the required libraries using pip:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Conclusion

This project provides a basic demonstration of using machine learning for classification with a Random Forest model. By following these steps, you can classify Iris flowers based on their measurements and visualize important features.
```

### Explanation:

This explanation and code structure for your GitHub `README.md` file covers:

1. **Project Overview**: A description of the Iris flower classification project and its goals.
2. **Code Steps**: A step-by-step breakdown of each part of the code, including:
   - Importing necessary libraries.
   - Loading and preparing the dataset.
   - Training the Random Forest model.
   - Evaluating the model.
   - Visualizing feature importance.
   - Making predictions on new data.
3. **Requirements**: Specifies the necessary Python libraries and how to install them.
4. **Conclusion**: Summarizes what the project accomplishes and what the user can expect.
