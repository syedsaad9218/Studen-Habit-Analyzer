import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import os

def main():
    # load dataset
    df = pd.read_csv('data/student_habits.csv')

    # feature and target
    X = df.drop('performance', axis=1)
    y = df['performance']

    #encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #model 1: logistic regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)

    print("Logistic Regression results:")
    y_pred_lr = log_reg.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_lr, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

    #model 2: random forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    print("Random Forest results:")
    y_pred_rf = rf.predict(X_test)
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': rf, 'scaler': scaler, 'label_encoder': le}, 'models/model_v1.joblib')

if __name__ == "__main__":
    main()