import pickle

# Load feature names
with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("Feature Names Saved in Model:")
print("="*50)
for i, name in enumerate(feature_names, 1):
    print(f"  {i:2d}. {name}")

print(f"\nTotal Features: {len(feature_names)}")

# Load model info
with open('models/model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

print(f"\nModel Info:")
print(f"  Model    : {model_info['model_name']}")
print(f"  Accuracy : {model_info['accuracy']:.4f}")

# Load scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print(f"\nScaler Info:")
print(f"  Type     : {type(scaler)}")
print(f"  Features : {scaler.n_features_in_}")