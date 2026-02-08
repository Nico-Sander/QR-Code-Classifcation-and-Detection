import pandas as pd

# Lade das Logfile
df = pd.read_csv('runs/run_01_baseline_cnn_20251231_124722/training_log.csv')

# Finde die Zeile mit dem minimalen Validation Loss
best_epoch = df.loc[df['val_loss'].idxmin()]

print("--- WERTE FÜR DEINE TABELLE ---")
print(f"Epoch:      {int(best_epoch['epoch'] + 1)}")
print(f"Train Loss: {best_epoch['loss']:.4f}")
print(f"Val Loss:   {best_epoch['val_loss']:.4f}")
print(f"Train Acc:  {best_epoch['accuracy']:.2%}")
print(f"Val Acc:    {best_epoch['val_accuracy']:.2%}")