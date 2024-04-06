import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class GLFS25Predictor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.label_mapping = {1: 1, 2: 2, 3: 3}
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        self.dt_model = None
        self.train_model()

    def train_model(self):
        # Load the provided dataset
        df = pd.read_csv(self.dataset_path)

        # Convert float columns to integers
        df = df.astype({'Age': int, 'SingleLegStandingTime': int, 'GripStrength': int, 'TUG_Result': int, 'GLFS25': int})

        # Extract relevant columns from the dataset
        selected_attributes = ['Age', 'SingleLegStandingTime', 'GripStrength', 'TUG_Result', 'GLFS25']
        selected_df = df[selected_attributes]

        # Map GLFS25 values to numerical labels
        selected_df['GLFS25'] = selected_df['GLFS25'].map(self.label_mapping)

        # Split the dataset into features (X) and target labels (y)
        X = selected_df.drop('GLFS25', axis=1)
        y = selected_df['GLFS25']

        # Train a DecisionTreeClassifier
        self.dt_model = DecisionTreeClassifier(random_state=42)
        self.dt_model.fit(X, y)

    def predict_glfs25(self, age, slst, grip_strength, tug_test_result):
        # Make a prediction
        input_data = pd.DataFrame([[age, slst, grip_strength, tug_test_result]],
                                  columns=['Age', 'SingleLegStandingTime', 'GripStrength', 'TUG_Result'])
        predicted_glfs = self.dt_model.predict(input_data)[0]

        # Map numerical label back to GLFS25 values
        predicted_glfs_label = self.reverse_label_mapping.get(predicted_glfs, "Unknown")

        return predicted_glfs_label

# Create an instance of the GLFS25Predictor class
dataset_path = 'locodataset.csv'  
glfs_predictor = GLFS25Predictor(dataset_path)

# Get input values from the user
age = int(input("Enter Age in years: "))
slst = int(input("Enter Single-Leg Standing Time in seconds: "))
grip_strength = int(input("Enter Grip Strength in kg: "))
tug_test_result = int(input("Enter Timed Up and Go (TUG) test result in seconds: "))

# Make a prediction using the trained model
predicted_glfs = glfs_predictor.predict_glfs25(age, slst, grip_strength, tug_test_result)

print(f"The predicted GLFS25 value is: {predicted_glfs}")
