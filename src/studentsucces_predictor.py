import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv('University_Student_Dataset_Final_with_Pass_Fail.csv')
print(df.head())
print(df.info())
print(df.describe())
print("null values in each column:")
print(df.isnull().sum())
df = df.copy()

def main_menu(model,scaler,input_df):
    print("------------------------------")
    print("wlecome to student success predictor")
    print("------------------------------")
    print("select the option 1. power mode 🚀 2.quick mode 💻 , 3.exit")
    user_choice = input("enter your choice: (1/2/3): ")
    if user_choice == '1':
        print("You are now in Power Mode 💪")
        power_mode(model,scaler,input_df)
    elif user_choice == '2':
        print("you are in Quick Mode ⚡")
        quick_mode(model,scaler,input_df)
    elif user_choice == '3':
        print("Exiting the program. Goodbye!👋😄")
        exit()

def power_mode(model, scaler, df):
    print("\nYou are in Power Mode 💪")
    print("Please enter all features used in the model:")

    input_array = np.zeros(df.shape[1])
    for i,feature in enumerate(df.columns):
        try:
            val = float(input(f"{feature}: "))
            input_array[i] = val
        except ValueError:
            print(f"Invalid input for {feature}. Expected a number. Setting it to 0.")

    user_inputs_array = input_array.reshape(1, -1)
    user_inputs_scaled = scaler.transform(user_inputs_array)
    prediction = model.predict(user_inputs_scaled)

    print("\n🎓 Prediction Result:")
    if prediction[0] == 1:
        print("✅ You are likely to Pass! 🎉")
    else:
        print("❌ You might Fail. Don't give up, keep improving! 💪")


def quick_mode(model, scaler, df):
    print("\nYou are in Quick Mode ⚡")
    print("Please enter the following Information:")
    
    default_features = [
        "Previous_Grades",
        "Attendance",
        "Class_Participation",
        "Motivation",
        "Parental_Involvement",
        "Stress_Levels",
        "Professor_Quality",
    ]
    input_array = np.zeros(df.shape[1])

    for feature in default_features:
        if feature in df.columns : 
            try:
             val = float(input(f"Enter value for {feature}: "))
             feature_index = df.columns.get_loc(feature)
             input_array[feature_index] = val
            except ValueError:
                print(f"Invalid input for {feature}. Expected a number. Setting it to 0.")
        else:
            print(f"⚠️ Warning: {feature} not found in model input columns. Skipping.")

    user_inputs_array = input_array.reshape(1, -1)
    user_inputs_scaled = scaler.transform(user_inputs_array)

    prediction = model.predict(user_inputs_scaled)
    
    print("\n🎓 Prediction Result:")
    if prediction[0] == 1:
        print("✅ You are likely to Pass! 🎉")
    elif prediction[0] == 0:
        print("❌ You might Fail. Don't give up, keep improving! 💪")

ordinal_features = [
    "Previous_Grades",
    "Parental_Education",
    "Attendance",
    "Class_Participation",
    "Financial_Status",
    "Parental_Involvement",
    "Motivation",
    "Self_Esteem",
    "Stress_Levels",
    "School_Environment",
    "Professor_Quality",
    "Sleep_Patterns",
    "Nutrition",
    "Physical_Activity",
    "Lack_of_Interest"
]
nominal_features = [
    "Gender",
    "Major",
    "School_Type",
    "Educational_Resources",
    "Extracurricular_Activities",
    "Screen_Time",  
    "Educational_Tech_Use",
    "Peer_Group",
    "Bullying",
    "Study_Space",
    "Learning_Style",
    "Tutoring",
    "Mentoring",
    "Sports_Participation"
]

le = LabelEncoder()
df["Pass_Fail"] = le.fit_transform(df["Pass_Fail"])
for feature in ordinal_features:
    
    df[feature] = le.fit_transform(df[feature])

df = pd.get_dummies(df,columns=nominal_features, drop_first=True)
print(df)

x = df.drop("Pass_Fail", axis=1)
y = df["Pass_Fail"]

x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
StandardScaled = StandardScaler()
x_train = StandardScaled.fit_transform(x_train)
x_test = StandardScaled.transform(x_test)

model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("----------Classification Report:")
classification_rep = classification_report(y_test,y_pred)
print(classification_rep)
print("----------Confusion Matrix:")
confusion_mat = confusion_matrix(y_test,y_pred)
print(confusion_mat)
plt.figure(figsize= (8,6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Fail', 'Predicted Pass'], yticklabels=['Actual Fail', 'Actual Pass'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
print()
print("___________Predict your success here_________________")
main_menu(model, StandardScaled, x)