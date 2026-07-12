# Data

This directory contains the dataset files used by the ET-PRL experiment workflows.

Expected layout:

```text
data/
  raw_data.csv
  action_all.npy
  coeff_date.npy
  lstm/
    train.csv
    val.csv
    test.csv
    normalizer.json
  dqn/
    train_data.csv
    val_data.csv
    test_data.csv
    cl_next_predictions.csv
    value_gate_dataset.csv
    value_gate_dataset.meta.json
  value_gate_workbench/
    value_gate_dataset.csv
    value_gate_dataset.meta.json
```
