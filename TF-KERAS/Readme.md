# TF/Keras notes

# Callbacks

- Define callbacks to do the following tasks:
  - Reduce learning rate when validation loss does not improve for 5 continuous epochs.
  - Early stopping when validation loss does not improve for 10 continuous epochs.
  - When there is a validation loss improvement, save the model's weight.

```
callbacks = [
	ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
	EarlyStopping(patience=10, verbose=1),
	ModelCheckpoint('model_name.h5', verbose=1, save_best_only=True, save_weights_only=True)
]
```