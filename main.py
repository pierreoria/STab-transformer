import os
import gc
from copy import deepcopy
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime

import kagglehub
from kagglehub import KaggleDatasetAdapter

from keras.models import Sequential
from keras.layers import Dense, Input
#from keras.callbacks import EarlyStopping
from keras4torch.callbacks import EarlyStopping
from keras.metrics import MeanSquaredError, BinaryCrossentropy

import tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras import Sequential, layers, regularizers, optimizers
import optuna
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import optuna

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, average_precision_score
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier

# Bibliotecas principais
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import roc_auc_score, classification_report

# Keras4Torch
import keras4torch
from keras4torch.callbacks import ModelCheckpoint, LRScheduler
from keras4torch.callbacks import Callback


# STab
from STab import mainmodel, LWTA, Gsoftmax, Num_Cat

import scikitplot as skplt
import matplotlib
import matplotlib.pyplot as plt

import math
import torch
import numpy as np
from sklearn.metrics import classification_report
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss


RANDOM_STATE = 42
EPOCHS = 10000

file_path = "customer_churn_telecom_services.csv"

df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "kapturovalexander/customers-churned-in-telecom-services",
  file_path,
)

PREDICTABLE_CLASSES = df['Churn'].unique().tolist()

target_map = {'No': 0, 'Yes': 1}

def split_class(df_class):
    # Separação entre X e Y
    Y = df_class['Churn'].map(target_map)
    X = df_class.drop('Churn', axis=1)

    # Código-base para separação dos conjuntos
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, Y, test_size=0.5, stratify=Y, random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )

    # Nova união entre X e y de cada conjunto, para realizar técnicas de balanceamento
    df_train = X_train.copy()
    df_train['Churn'] = y_train

    df_val = X_val.copy()
    df_val['Churn'] = y_val

    df_test = X_test.copy()
    df_test['Churn'] = y_test

    return df_train, df_val, df_test

def split_and_balance(df):
    df_true = df[df['Churn'] == 'Yes']
    df_false = df[df['Churn'] == 'No']

    c1_train, c1_val, c1_test = split_class(df_true)
    c0_train, c0_val, c0_test = split_class(df_false)

    min_train_len = min(len(c1_train), len(c0_train))
    max_train_len = max(len(c1_train), len(c0_train))
    min_val_len = min(len(c1_val), len(c0_val))
    max_val_len = max(len(c1_val), len(c0_val))

    if len(c1_train) < len(c0_train):
        c1_train = resample(c1_train, replace=True, n_samples=max_train_len, random_state=RANDOM_STATE)
        c1_val = resample(c1_val, replace=True, n_samples=max_val_len, random_state=RANDOM_STATE)
    else:
        c0_train = resample(c0_train, replace=True, n_samples=max_train_len, random_state=RANDOM_STATE)
        c0_val = resample(c0_val, replace=True, n_samples=max_val_len, random_state=RANDOM_STATE)

    train = pd.concat([c1_train, c0_train]).sample(frac=1, random_state=RANDOM_STATE)
    val = pd.concat([c1_val, c0_val]).sample(frac=1, random_state=RANDOM_STATE)
    test = pd.concat([c1_test, c0_test]).sample(frac=1, random_state=RANDOM_STATE)

    return train, val, test

df_train, df_val, df_test = split_and_balance(df)

y_train = df_train['Churn']
X_train = df_train.drop('Churn', axis=1)

y_val = df_val['Churn']
X_val = df_val.drop('Churn', axis=1)

y_test = df_test['Churn']
X_test = df_test.drop('Churn', axis=1)

def fill_absent_features(df, median_num=None, median_cat=None):
  total_rows = len(df)

  for col in df.columns:
      null_count = df[col].isnull().sum()
      if null_count > 0:
          pct = 100 * null_count / total_rows
          if df[col].dtype in ['int64', 'float64']:
              if median_num is None:
                  median_num = df[col].median()
              df.loc[:, col] = df[col].fillna(median_num)
              print(f"{col}: {null_count} valores ausentes ({pct:.2f}% do dataset) preenchidos com mediana {median_num}")
          else:
              if median_cat is None:
                  median_cat = df[col].median()
              df.loc[:, col] = df[col].fillna(median_cat)
              print(f"{col}: {null_count} valores ausentes ({pct:.2f}% do dataset) preenchidos com moda '{median_cat}'")

  return df, median_num, median_cat

def scale_columns(df, scaler=None):
  norm_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

  if scaler is None:
    scaler = MinMaxScaler()
    df[norm_cols] = scaler.fit_transform(df[norm_cols])
  else:
    df[norm_cols] = scaler.transform(df[norm_cols])

  return df, scaler

def boolean_to_int(df):
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df

def preprocess_df(df_train, df_val, df_test):
  df_train, median_num, median_cat = fill_absent_features(df_train)
  df_val, _, _ = fill_absent_features(df_val, median_num, median_cat)
  df_test, _, _ = fill_absent_features(df_test, median_num, median_cat)

  df_train, train_scaler = scale_columns(df_train)
  df_val, _ = scale_columns(df_val, train_scaler)
  df_test, _ = scale_columns(df_test, train_scaler)

  df_train = pd.get_dummies(df_train)
  df_val = pd.get_dummies(df_val)
  df_test = pd.get_dummies(df_test)

  df_train = boolean_to_int(df_train)
  df_val = boolean_to_int(df_val)
  df_test = boolean_to_int(df_test)

  return df_train, df_val, df_test


X_train, X_val, X_test = preprocess_df(X_train, X_val, X_test)

INPUT_DIM = X_train.shape[1]
################### ACABOU PREPROCESSAMENTO

def train_stab_classifier(
        classifier,
        epochs,
        batch_size,
        best_params,
        X_train_local,
        X_val_local,
        y_train_local,
        y_val_local,
        patience_epochs=10):

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay']
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    classifier.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'loss': [],
        'acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        # Train one epoch manually using fit_dl with epochs=1
        train_res = classifier.fit(
            X_train_local,
            y_train_local,
            batch_size=batch_size,
            epochs=1,            # only 1 epoch at a time
            validation_data=(X_val_local, y_val_local),
            verbose=1
        )

        print(train_res.columns)

        loss = train_res['loss'].iloc[-1]
        val_loss = train_res['val_loss'].iloc[-1]
        acc = train_res['acc'].iloc[-1]          # or 'accuracy' depending on output col name
        val_acc = train_res['val_acc'].iloc[-1]        # or 'val_accuracy'


        history['loss'].append(loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - train_loss: {loss:.4f}, val_loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_epochs:
                print(f"Early stopping at epoch {epoch+1} (no val_loss improvement for {patience_epochs} epochs)")
                break

    return history

def plot_confusion_matrix(y_true, y_pred):
  cm = confusion_matrix(y_true, y_pred)
  cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

  labels = np.array([
      [f"{count}\n({perc:.1f}%)" if not np.isnan(perc) else f"{count}\n(0.0%)"
        for count, perc in zip(row_counts, row_percents)]
      for row_counts, row_percents in zip(cm, cm_percentage)
  ])

  plt.figure(figsize=(6, 5))
  sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", xticklabels=PREDICTABLE_CLASSES, yticklabels=PREDICTABLE_CLASSES)
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.title("Confusion Matrix")
  plt.tight_layout()
  plt.show()

def plot_training_error_curves(history: dict):
  """Função para plotar as curvas de erro do treinamento da rede neural.

  Argumento(s):
  history -- Objeto retornado pela função fit do keras.

  Retorno:
  A função gera o gráfico do treino da rede e retorna None.
  """
  train_loss = history['loss']
  val_loss = history['val_loss']

  fig, ax = plt.subplots()
  ax.plot(train_loss, label='Train')
  ax.plot(val_loss, label='Validation')
  ax.set(title='Training and Validation Error Curves', xlabel='Epochs', ylabel='Loss (Cross-Entropy)')
  ax.legend()
  plt.show()

def extract_final_losses(history: dict):
  """Função para extrair o melhor loss de treino e validação.

  Argumento(s):
  history -- Objeto retornado pela função fit do keras.

  Retorno:
  Dicionário contendo o melhor loss de treino e de validação baseado
  no menor loss de validação.
  """
  train_loss = history['loss']
  val_loss = history['val_loss']
  idx_min_val_loss = np.argmin(val_loss)
  return {'train_loss': train_loss[idx_min_val_loss], 'val_loss': val_loss[idx_min_val_loss]}

def compute_performance_metrics(y, y_pred_class, y_pred_scores):
  accuracy = accuracy_score(y, y_pred_class)
  recall = recall_score(y, y_pred_class)
  precision = precision_score(y, y_pred_class)
  f1 = f1_score(y, y_pred_class)
  performance_metrics = (accuracy, recall, precision, f1)


  y_pred_scores = y_pred_scores[:, 1]

  _, _, _, ks, _, _ = skplt.helpers.binary_ks_curve(y, y_pred_scores)

  aucroc = roc_auc_score(y, y_pred_scores)
  aupr = average_precision_score(y, y_pred_scores)
  performance_metrics = performance_metrics + (aucroc, aupr)

  metrics = {
    'ks': ks,
    'accuracy': accuracy,
    'recall': recall,
    'precision': precision,
    'f1': f1,
    'aucroc': aucroc,
    'aupr': aupr,
  }

  for metric, value in metrics.items():
    print("{:<14}{:.4f}".format(metric + ":", float(value)))

  return ks, accuracy, recall, precision, f1, aucroc, aupr

 # ==== KS Plot Function ====
def plot_ks(y, y_pred_scores, save_path=None):
    if y_pred_scores.ndim == 1:
      y_pred_scores = np.column_stack((1 - y_pred_scores, y_pred_scores))
    skplt.metrics.plot_ks_statistic(y, y_pred_scores)
    if save_path:
      plt.savefig(save_path, bbox_inches='tight')
      print(f"KS plot saved to {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    labels = np.array([
       [f"{count}\n({perc:.1f}%)" if not np.isnan(perc) else f"{count}\n(0.0%)"
        for count, perc in zip(row_counts, row_percents)]
       for row_counts, row_percents in zip(cm, cm_percentage)
    ])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues",
        xticklabels=PREDICTABLE_CLASSES, yticklabels=PREDICTABLE_CLASSES
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
       plt.savefig(save_path, bbox_inches='tight')
       print(f"Confusion matrix saved to {save_path}")
    plt.close()

def plot_aucroc(y, y_pred_scores):
  y_pred_scores = y_pred_scores[:, 1]

  fpr, tpr, thresholds = roc_curve(y, y_pred_scores)
  roc_auc = roc_auc_score(y, y_pred_scores)

  plt.figure()
  plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlabel("FPR")
  plt.ylabel("TPR")
  plt.title("Curva ROC")
  plt.legend(loc="lower right")
  plt.show()






















# modelo com melhores parametros

Or_model = mainmodel.MainModel(
    categories=[],                   # no categorical embeddings, already one-hot
    num_continuous=45,              # all columns treated as numerical
    dim=16,                         # embedding dimension
    depth=2,                        # transformer layers
    heads=4,                        # attention heads
    dim_head=16,                    # per-head dim
    dim_out=1,                      # output dimension
    attn_dropout=0.1,
    ff_dropout=0.1,
    U=2,
    cases=16
)

model_wrapper = Num_Cat(Or_model, num_number=45, classes=1, Sample_size=16)
model = keras4torch.Model(model_wrapper).build([[45]])





# ----- Convert input features to float32 numpy arrays -----
X_train_stab = X_train.to_numpy().astype(np.float32)
X_val_stab = X_val.to_numpy().astype(np.float32)
X_test_stab = X_test.to_numpy().astype(np.float32)

# ----- Convert labels to float32 tensors (required for BCEWithLogitsLoss) -----
y_train_stab = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)
y_val_stab = torch.tensor(y_val.to_numpy(), dtype=torch.float32).unsqueeze(1)
y_test_stab = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1)

# ----- Define optimizer and loss -----
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = BCEWithLogitsLoss()

def objective(trial):

    # Sugestões de hiperparâmetros
    dim = trial.suggest_categorical('dim', [8, 16, 32, 64])
    depth = trial.suggest_int('depth', 1, 4)
    heads = trial.suggest_categorical('heads', [2, 4, 8])
    attn_dropout = trial.suggest_float('attn_dropout', 0.0, 0.5)
    ff_dropout = trial.suggest_float('ff_dropout', 0.0, 0.5)
    U = trial.suggest_int('U', 1, 4)
    cases = trial.suggest_categorical('cases', [8, 16, 32])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    sample_size = trial.suggest_categorical('sample_size', [8, 16, 32])

    # Cria o modelo
    Or_model = mainmodel.MainModel(
        categories=[],
        num_continuous=45,
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=16,
        dim_out=1,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        U=U,
        cases=cases
    )
    model_wrapper = Num_Cat(Or_model, num_number=45, classes=1, Sample_size=sample_size)
    model = keras4torch.Model(model_wrapper).build([[45]])

    # Configura otimizador e loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['f1']) 

    # Treina
    model.fit([X_train_stab], y_train_stab,
              epochs=10,
              batch_size=batch_size,
              validation_data=([X_val_stab], y_val_stab),
              verbose=2)

    # Predição na validação para AUC
    val_logits = model.predict([X_val_stab])
    val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy().squeeze()
    val_auc = roc_auc_score(y_val_stab.numpy(), val_probs)

    print(f'Val AUC: {val_auc:.4f}')

    return val_auc

#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=10)

# Save best results to a text file
#with open("best_params.txt", "w") as f:
#    f.write(f"Best parameters: {study.best_params}\n")
#    f.write(f"Best AUC: {study.best_value}\n")

#print("Best parameters and AUC written to best_params.txt")

##################################################

















############################################### Definição do modelo

# ==== Best params from Optuna ====
#best_params = {
#    'dim': 8,
#    'depth': 2,
#    'heads': 4,
#    'attn_dropout': 0.08038815261978655,
#    'ff_dropout': 0.2537406306674858,
#    'U': 1,
#    'cases': 16,
#    'lr': 0.008324262423344119,
#    'weight_decay': 7.291453787210219e-05,
#    'batch_size': 16,
#    'sample_size': 32
#}



# ==== Create model ====
#Or_model = mainmodel.MainModel(
#    categories=[],
#    num_continuous=45,
#    dim=best_params['dim'],
#    depth=best_params['depth'],
#    heads=best_params['heads'],
#    dim_head=16,
#    dim_out=1,
#    attn_dropout=best_params['attn_dropout'],
#    ff_dropout=best_params['ff_dropout'],
#    U=best_params['U'],
#    cases=best_params['cases']
#)

#model_wrapper = Num_Cat(
#    Or_model,
#    num_number=45,
#    classes=1,
#    Sample_size=best_params['sample_size']
#)

#stab_classifier = keras4torch.Model(model_wrapper).build([[45]])

## modelo default abaixo:
Or_model = mainmodel.MainModel(
    categories=[],                   # no categorical embeddings, already one-hot
    num_continuous=45,              # all columns treated as numerical
    dim=16,                         # embedding dimension
    depth=2,                        # transformer layers
    heads=4,                        # attention heads
    dim_head=16,                    # per-head dim
    dim_out=1,                      # output dimension
    attn_dropout=0.1,
    ff_dropout=0.1,
    U=2,
    cases=16
)

default_params = {
    'dim': 16,
    'depth': 2,
    'heads': 4,
    'attn_dropout': 0.1,
    'ff_dropout': 0.1,
    'U': 2,
    'cases': 16,
    'lr': 0.001,
    'weight_decay': 0.0001,
    'batch_size': 16,
    'sample_size': 16
}

model_wrapper = Num_Cat(Or_model, num_number=45, classes=1, Sample_size=16)
stab_classifier = keras4torch.Model(model_wrapper).build([[45]])



#################################################### Treinamento
stab_history = train_stab_classifier(
    classifier=stab_classifier,
    epochs=50,
    batch_size=16,
    best_params=default_params,
    X_train_local=X_train_stab,
    X_val_local=X_val_stab,
    y_train_local=y_train_stab,
    y_val_local=y_val_stab,
    patience_epochs=15
)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Save training error curves plot
ts = timestamp()
plot_training_error_curves(stab_history)
plt.savefig(f"training_error_curves_{ts}.png")
plt.close()

# Get raw predictions (logits)
#logits = stab_classifier.predict(X_test_stab)  # Shape: [n_samples, 1] or [n_samples]

# Apply sigmoid to get probabilities for class 1
#y_pred_proba = torch.sigmoid(torch.tensor(logits)).numpy()  # Shape: [n_samples]

# Stack probabilities for class 0 and class 1
#y_pred_proba_combined = np.column_stack([1 - y_pred_proba, y_pred_proba])  # Shape: [n_samples, 2]

########################
# Get raw predictions (logits) from your model
logits = stab_classifier.predict(X_test_stab)  # Shape: [n_samples, 1] or [n_samples]

# Convert logits to probabilities for class 1 (binary)
y_pred_proba = torch.sigmoid(torch.tensor(logits)).numpy().flatten()  # Ensure shape [n_samples]

y_pred_proba_combined = np.column_stack([1 - y_pred_proba, y_pred_proba])  # Shape: [n_samples, 2]

# Plot KS
ks_filename = f"ks_plot_{timestamp()}.png"
plot_ks(y_test_stab.numpy(), y_pred_proba_combined, save_path=ks_filename)


# Define thresholds
thresholds = np.arange(0.1, 1.0, 0.1)

for thr in thresholds:
    # Predict classes using threshold (same as before)
    stab_y_pred_class = np.where(y_pred_proba >= thr, 1, 0)  # Use y_pred_proba here!

    # Save classification report
    report = classification_report(y_test_stab.numpy(), stab_y_pred_class)
    with open(f"classification_report_thr_{thr:.1f}_{timestamp()}.txt", "w") as f:
        f.write(f"Threshold: {thr:.1f}\n\n")
        f.write(report)  # Fixed typo (was 'report' before)

    # Plot confusion matrix
    cm_filename = f"confusion_matrix_thr_{thr:.1f}_{timestamp()}.png"
    plot_confusion_matrix(y_test_stab.numpy(), stab_y_pred_class, save_path=cm_filename)
    plt.close()



# Predict scores once
#stab_y_pred_scores = stab_classifier.predict_proba(X_test_stab)
#stab_y_pred_scores_0 = 1 - stab_y_pred_scores
#stab_y_pred_scores_combined = np.concatenate([stab_y_pred_scores_0, stab_y_pred_scores], axis=1)

#thresholds = np.arange(0.1, 1.0, 0.1)

#for thr in thresholds:
#    # Predict classes using threshold
#    stab_y_pred_class = np.where(stab_y_pred_scores >= thr, 1, 0)
#
#    # Save classification report
#    report = classification_report(y_test_stab.numpy(), stab_y_pred_class)
#    with open(f"classification_report_thr_{thr:.1f}_{timestamp()}.txt", "w") as f:
#        f.write(f"Threshold: {thr:.1f}\n\n")
#        f.write(report)
#
#    # Plot confusion matrix, save with timestamp
#    cm_filename = f"confusion_matrix_thr_{thr:.1f}_{timestamp()}.png"
#    plot_confusion_matrix(y_test_stab.numpy(), stab_y_pred_class, save_path=cm_filename)
#    plt.close()



#stab_y_pred_scores = stab_classifier.predict(X_test_stab)
#stab_y_pred_class = np.where(stab_y_pred_scores >= 0.5, 1, 0)
#stab_y_pred_scores_0 = 1 - stab_y_pred_scores
#stab_y_pred_scores = np.concatenate([stab_y_pred_scores_0, stab_y_pred_scores], axis=1)

#print("\nClassification Report:")
#print(classification_report(y_test_stab.numpy(), stab_y_pred_class))

#plot_ks(y_test_stab.numpy(), stab_y_pred_scores, save_path="ks_plot.png")
#plot_confusion_matrix(y_test_stab.numpy(), stab_y_pred_class)

# # ==== Predictions & Metrics ====
# test_logits = model.predict([X_test_stab])
# test_probs = torch.sigmoid(torch.tensor(test_logits)).numpy().squeeze()

# test_auc = roc_auc_score(y_test_stab.numpy(), test_probs)
# print(f"Validation AUC: {test_auc:.4f}")

# y_pred_labels = (test_probs >= 0.5).astype(int)










######################################################### Treinamento normal sem early stopping

# # ==== Compile ====
# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=best_params['lr'],
#     weight_decay=best_params['weight_decay']
# )
# loss_fn = torch.nn.BCEWithLogitsLoss()

# model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# # ==== Train ====
# model.fit(
#     [X_train_stab],
#     y_train_stab,
#     epochs=20,
#     batch_size=best_params['batch_size'],
#     validation_data=([X_val_stab], y_val_stab),
#     verbose=2
# )
