import pandas as pd

# Load current data
df = pd.read_csv('data/conversations/raw/fraud_processed.csv')

# Create new dataframe with required columns
processed_df = pd.DataFrame({
    'TransactionID': df.apply(lambda row: ' '.join([str(row[col]) for col in df.columns if col != 'isFraud']), axis=1),
    'is_fraud': df['isFraud']
})

# Save new format
processed_df.to_csv('data/conversations/raw/fraud_text_format.csv', index=False)