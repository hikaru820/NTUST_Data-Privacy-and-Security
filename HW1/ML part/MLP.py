# MLP.py
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from datetime import datetime
import copy
import joblib
import os

from DataConversion import x_train, x_test, y_train, y_test

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

OUTPUT_MODEL = True
MAX_ITERATION = 100
best_acc   = 0
best_auc   = 0
best_model = None
best_epoch = 0

mlp = MLPClassifier(
    hidden_layer_sizes = (64, 32),
    activation         = 'tanh',
    solver             = 'adam',
    max_iter           = 1,         
    warm_start         = True,      # keeps weights between partial fits
    random_state       = 42,
    alpha = 0.01
)

n_epochs = MAX_ITERATION
for epoch in range(1, n_epochs + 1):
    mlp.fit(x_train, y_train)

    train_acc = accuracy_score(y_train, mlp.predict(x_train))
    test_acc  = accuracy_score(y_test,  mlp.predict(x_test))
    auc       = roc_auc_score(y_test, mlp.predict_proba(x_test)[:, 1])

    if epoch % 10 == 0:
        print(f"Iteration {epoch}/{n_epochs} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc   = auc
        best_acc   = test_acc
        best_epoch = epoch
        best_model = copy.deepcopy(mlp)
        
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if OUTPUT_MODEL:
    os.makedirs('MLPmodels', exist_ok=True)
    filename = f"MLPmodels/MLPmodel_AUC-{best_auc:.4f}_ACC-{best_acc:.4f}_{timestamp}.pkl"
    joblib.dump(best_model, filename)

YELLOW = '\x1b[33m'
WHITE = '\x1b[37m'
print(f"{YELLOW}Best model saved at epoch {best_epoch} with AUC: {best_auc:.4f} and accuracy: {best_acc:.4f}{WHITE}")

y_pred     = best_model.predict(x_test)
y_prob     = best_model.predict_proba(x_test)[:, 1]

accuracy   = accuracy_score(y_test, y_pred)
misclass   = 1 - accuracy
precision  = precision_score(y_test, y_pred)
recall     = recall_score(y_test, y_pred)
auc        = roc_auc_score(y_test, y_prob)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Misclassification: {misclass:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"AUC:               {auc:.4f}")

if OUTPUT_MODEL:
    os.makedirs('MLPlogs', exist_ok=True)
    logfile = f"MLPlogs/MLPresult_{timestamp}.txt"

    with open(logfile, 'w') as f:
        f.write(f"=== MLP Results ===\n")
        f.write(f"Timestamp:         {timestamp}\n")
        f.write(f"Best Epoch:        {best_epoch}/{MAX_ITERATION}\n")
        f.write(f"\n--- Model Config ---\n")
        f.write(f"Hidden Layers:     {mlp.hidden_layer_sizes}\n")
        f.write(f"Activation:        {mlp.activation}\n")
        f.write(f"Alpha:             {mlp.alpha}\n")
        f.write(f"\n--- Metrics ---\n")
        f.write(f"AUC:               {auc:.4f}\n")
        f.write(f"Accuracy:          {accuracy:.4f}\n")
        f.write(f"Misclassification: {misclass:.4f}\n")
        f.write(f"Precision:         {precision:.4f}\n")
        f.write(f"Recall:            {recall:.4f}\n")

    print(f"Results logged to '{logfile}'")