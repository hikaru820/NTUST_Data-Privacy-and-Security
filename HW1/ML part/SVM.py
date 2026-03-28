# Model.py
from sklearn.linear_model import SGDClassifier
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import copy
import joblib
import os

from DataConversion import x_train, x_test, y_train, y_test

OUTPUT_MODEL = True
DYNAMIC_SCHEDULER = True
MAX_ITERATION = 300

best_acc = 0
best_auc = 0 # Will determine who is the best model
best_model = None
best_epoch = 0



if(DYNAMIC_SCHEDULER):
    sgd = SGDClassifier(loss = 'hinge', random_state = 42,
                        learning_rate = 'invscaling', eta0 = 0.005, power_t = 0.1)
else:
    sgd = SGDClassifier(loss = 'hinge', random_state = 42)

n_epochs = MAX_ITERATION
for epoch in range(1, n_epochs + 1):
    sgd.partial_fit(x_train, y_train, classes=[0, 1])
    
    train_acc = accuracy_score(y_train, sgd.predict(x_train))
    test_acc  = accuracy_score(y_test,  sgd.predict(x_test))
    auc = roc_auc_score(y_test, sgd.decision_function(x_test))
    
    if epoch % 10 == 0:
        print(f"Iteration {epoch}/{n_epochs} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | AUC: {auc:.4f}")
        
    if auc > best_auc:
        best_auc = auc
        best_acc = test_acc
        best_epoch = epoch
        best_model = copy.deepcopy(sgd)
        
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if(OUTPUT_MODEL):
    os.makedirs('SVMmodels', exist_ok=True)
    filename = f"SVMmodels/SVMmodel_AUC-{best_auc:.4f}_ACC-{best_acc:.4f}_{timestamp}.pkl"
    joblib.dump(best_model, filename)
    
YELLOW = '\x1b[33m'
WHITE = '\x1b[37m'
print(f"{YELLOW}Best model saved at epoch {best_epoch} with AUC: {best_auc:.4f} and accuracy: {best_acc:.4f}{WHITE}")

y_pred     = best_model.predict(x_test)
y_prob     = sgd.decision_function(x_test)

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
    os.makedirs('SVMlogs', exist_ok=True)
    logfile = f"SVMlogs/SVMresult_{timestamp}.txt"
    
    with open(logfile, 'w') as f:
        f.write(f"=== SVM Results ===\n")
        f.write(f"Timestamp:         {timestamp}\n")
        f.write(f"Best Epoch:        {best_epoch}/{MAX_ITERATION}\n")
        f.write(f"Dynamic Scheduler: {DYNAMIC_SCHEDULER}\n")
        f.write(f"\n--- Model Config ---\n")
        f.write(f"Learning Rate:     {sgd.learning_rate}\n")
        f.write(f"eta0:              {sgd.eta0}\n")
        f.write(f"\n--- Metrics ---\n")
        f.write(f"AUC:               {auc:.4f}\n")
        f.write(f"Accuracy:          {accuracy:.4f}\n")
        f.write(f"Misclassification: {misclass:.4f}\n")
        f.write(f"Precision:         {precision:.4f}\n")
        f.write(f"Recall:            {recall:.4f}\n")
    
    print(f"Results logged to '{logfile}'")