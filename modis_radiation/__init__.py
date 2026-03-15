"""
modis_radiation
===============
Sea-surface net-radiation inversion from MODIS TOA reflectance / brightness
temperature using a Set-Attention daily encoder followed by an
LSTM + Transformer time-series model.

Package layout
--------------
observation_encoder.py  – Set-attention encoder that reduces a variable number
                          of within-day MODIS swath observations to one fixed-
                          length vector.
model.py                – Full temporal model (Encoder → LSTM → Transformer).
dataset.py              – PyTorch Dataset / DataLoader helpers with time-based
                          train / validation / test splitting.
train.py                – End-to-end training and evaluation script.
"""
