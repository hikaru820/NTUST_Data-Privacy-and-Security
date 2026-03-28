# main.py

from DataConversion import load_original_data
from DataConversion_Kanon import load_anonymized
from SVM import train_svm
from MLP import train_mlp

K_VALUES = [2, 5, 10, 25, 50]

print("=== Training on Original Data ===")
x_train_orig, x_test_orig, y_train_orig, y_test_orig = load_original_data()

svm_results = {'original': train_svm(x_train_orig, x_test_orig, y_train_orig, y_test_orig, label='original')}
mlp_results = {'original': train_mlp(x_train_orig, x_test_orig, y_train_orig, y_test_orig, label='original')}

for k in K_VALUES:
    print(f"\n=== Training on K={k} Anonymized Data ===")
    x_train_anon, _, y_train_anon, _ = load_anonymized(k)  # only use train portion
    svm_results[k] = train_svm(x_train_anon, x_test_orig, y_train_anon, y_test_orig, label=f'k{k}')
    mlp_results[k] = train_mlp(x_train_anon, x_test_orig, y_train_anon, y_test_orig, label=f'k{k}')

# Final comparison summary
print("\n=== Results Summary ===")
print(f"{'Dataset':<12} {'SVM AUC':>8} {'SVM ACC':>8} {'MLP AUC':>8} {'MLP ACC':>8}")
for key in ['original'] + K_VALUES:
    print(f"{'K='+str(key) if key != 'original' else 'Original':<12} "
          f"{svm_results[key]['auc']:>8.4f} {svm_results[key]['accuracy']:>8.4f} "
          f"{mlp_results[key]['auc']:>8.4f} {mlp_results[key]['accuracy']:>8.4f}")