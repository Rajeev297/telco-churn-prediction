from src.preprocessing import DataPreprocessor
from src.data_loader import DataLoader

# Load data
df = DataLoader().load_sample_data()
df_copy = df.drop('Churn', axis=1).iloc[:100]

print('Before convert_numeric_columns:')
print(f'  TotalCharges dtype: {df_copy["TotalCharges"].dtype}')
print(f'  Sample: {df_copy["TotalCharges"].head()}')

# Create preprocessor and convert
prep = DataPreprocessor()
df_converted = prep.convert_numeric_columns(df_copy)

print('\nAfter convert_numeric_columns:')
print(f'  TotalCharges dtype: {df_converted["TotalCharges"].dtype}')
print(f'  Sample: {df_converted["TotalCharges"].head()}')
