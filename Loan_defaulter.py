import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


print("Current Working Directory:", os.getcwd())


file_path_train = r"C:\Users\krish\OneDrive\Desktop\loan defaulter\train.csv"
file_path_test = r"C:\Users\krish\OneDrive\Desktop\loan defaulter\test.csv"


try:
    train_data = pd.read_csv(file_path_train)
    test_data = pd.read_csv(file_path_test)
    print("Datasets loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please verify that the file paths are correct and the files exist.")


if 'train_data' in locals() and 'test_data' in locals():

    categorical_columns = ['Gender', 'Marital_Status', 'Employment_Status', 'Loan_Type']
    encoder = LabelEncoder()
    for col in categorical_columns:
        train_data[col] = encoder.fit_transform(train_data[col])
        test_data[col] = encoder.transform(test_data[col])

  
    numerical_columns = ['Age', 'Income', 'Debt_to_Income_Ratio', 'Credit_Score', 
                         'Loan_Amount', 'Interest_Rate', 'On_Time_Payments', 
                         'Missed_Payments', 'Outstanding_Balance']
    scaler = StandardScaler()
    train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
    test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

 
    X_train = train_data.drop('Default_Status', axis=1)
    y_train = train_data['Default_Status']
    X_test = test_data.drop('Default_Status', axis=1)
    y_test = test_data['Default_Status']

    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }


    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

   
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)

 
    y_pred = best_model.predict(X_test)

   
    classification_rep = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)


    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png') 
    plt.show()


    with open('loan_defaulter_results.txt', 'w') as f:
        f.write("Classification Report:\n")
        f.write(classification_rep)
        f.write("\nAccuracy: {:.4f}\n".format(accuracy))

    print("Confusion matrix image saved as 'confusion_matrix.png'")
    print("Results saved in 'loan_defaulter_results.txt'")

else:
    print("Datasets could not be loaded. Exiting.")
