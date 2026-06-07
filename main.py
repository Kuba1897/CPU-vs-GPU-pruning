from test import *

print("Running non-pruning test on GPU...")
non_pruning_test_gpu()

print("\nRunning unstructured pruning test on GPU...")
unstructured_pruning_test_gpu()

print("\nRunning structured pruning test on GPU...")
structured_pruning_test_gpu()

print("\nRunning non-pruning test on CPU...")
non_pruning_test_cpu()

print("\nRunning unstructured pruning test on CPU...")
unstructured_pruning_test_cpu()

print("\nRunning structured pruning test on CPU...")
structured_pruning_test_cpu()