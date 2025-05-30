
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from stepmix.stepmix import StepMix
from sklearn.metrics import confusion_matrix
import warnings

print("updated")
# Load and prepare data
def load_data_lca(lca_df):
    # data = lca_df
    data = lca_df.copy() 
    # Check for non-discriminative indicators
    for col in data.columns:
        data[col] = data[col].astype("category")
    for col in data.columns:
        if len(data[col].unique()) == 1:
            warnings.warn(f"Indicator '{col}' has no variation (all {data[col].iloc[0]}). It won't contribute to class separation.")
    
    return data

# Compute normalized entropy
def compute_entropy(probs):
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1).mean()
    normalized_entropy = entropy / np.log(probs.shape[1])
    return normalized_entropy

# Fit LCA model and evaluate
# Fit LCA model with cross-validation
def fit_lca(data, n_classes, n_init=20, max_iter=500, abs_tol=1e-6, rel_tol=1e-6):
    # Initialize model
    model = StepMix(
        n_components=n_classes,
        measurement='categorical',
        random_state=42,
        max_iter=max_iter,
        n_init=n_init,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        verbose=1
    )
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_log_likelihoods = []
    
    for train_idx, test_idx in kf.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        try:
            model.fit(train_data)
            ll = model.score(test_data)  # Log-likelihood on test set
            cv_log_likelihoods.append(ll)
        except Exception as e:
            print(f"CV fold failed: {e}")
            continue
    
    # Fit on full data
    try:
        model.fit(data)
    except Exception as e:
        print(f"Error during full fit: {e}")
        return None, None, None, None, None, None
    
    # Get assignments and probabilities
    assignments = model.predict(data)
    probs = model.predict_proba(data)
    
    # Compute metrics
    aic = model.aic(data)
    bic = model.bic(data)
    entropy = compute_entropy(probs)
    
    return model, assignments, probs, aic, bic, entropy, np.mean(cv_log_likelihoods)

