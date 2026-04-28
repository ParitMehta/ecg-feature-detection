metadata shape: (21799, 28)
Files on disk: 21799 of 21799 (100.0%). Dropped 0 missing.

Class counts:
diagnostic_superclass
NORM    9514
MI      5469
STTC    5235
CD      4898
HYP     2649

411 of 21799 records have zero superclass labels
sample shape: (5, 1000, 12)

Labels per record:
diagnostic_superclass
0      411
1    16244
2     4068
3      919
4      157

Per-fold class proportions:
      fold_1  fold_2  fold_3  fold_4  fold_5  fold_6  fold_7  fold_8  fold_9  fold_10
NORM   0.341   0.348   0.356   0.336   0.336   0.334   0.349   0.340   0.343    0.345
MI     0.199   0.194   0.190   0.200   0.201   0.202   0.198   0.196   0.194    0.197
STTC   0.191   0.189   0.185   0.191   0.190   0.190   0.187   0.187   0.190    0.187
CD     0.174   0.175   0.175   0.179   0.177   0.177   0.172   0.179   0.178    0.178
HYP    0.095   0.095   0.095   0.095   0.095   0.097   0.095   0.098   0.096    0.094

Sanity checks on 100 random records:
  NaNs present: False
  shape: (100, 1000, 12)
  value range: [-4.223, 3.850] (mV)
  per-lead std min: 0.1243