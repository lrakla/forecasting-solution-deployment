
# import lightning.pytorch as pl
# from lightning.pytorch.callbacks import EarlyStopping
# import pandas as pd
# import torch
# import pytorch_forecasting as ptf
# from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
# from pytorch_forecasting.data import NaNLabelEncoder
# from pytorch_forecasting.metrics import MAE, SMAPE, MultivariateNormalDistributionLoss
# from lightning.pytorch.tuner import Tuner

data_deepar = data.sort_values(by='Date')
data_deepar['time_idx'] = data_deepar.groupby('Date').ngroup()
data_deepar['IsHoliday'] = data_deepar['IsHoliday'].astype(str)
data_deepar['Type'] = data_deepar['Type'].astype(str)
max_encoder_length = 10
max_prediction_length = 3
training_cutoff = '2012-10-5'
min_encoder_length = 3
context_length = max_encoder_length
prediction_length = max_prediction_length
training_cutoff = data_deepar["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data_deepar[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Weekly_Sales",
    group_ids=["Store", "Dept"],
    static_categoricals=[
        "Store",
        "Dept",
        "IsHoliday",
        "Type"
    ], 
    categorical_encoders={
        'IsHoliday': NaNLabelEncoder(add_nan=True),
        'Type':NaNLabelEncoder(add_nan=True),
    },
     static_reals = ["Size"], # as we plan to forecast correlations, it is important to use series characteristics (e.g. a series identifier)
    time_varying_known_reals=["Temperature","Fuel_Price","CPI","Unemployment"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    min_encoder_length=min_encoder_length,
    allow_missing_timesteps=True,
    time_varying_unknown_reals=["Weekly_Sales"]
)

validation = TimeSeriesDataSet.from_dataset(training, data_deepar, min_prediction_idx= training_cutoff + 1 , stop_randomization=True)
batch_size = 3
# synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
)

baseline_predictions = Baseline().predict(val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
SMAPE()(baseline_predictions.output, baseline_predictions.y)

pl.seed_everything(42)


trainer = pl.Trainer(accelerator="cpu", gradient_clip_val=1e-1)
net = DeepAR.from_dataset(
    training,
    learning_rate=3e-2,
    hidden_size=30,
    rnn_layers=2,
    loss=MultivariateNormalDistributionLoss(rank=30),
    optimizer="Adam",
)
# find optimal learning rate


res = Tuner(trainer).lr_find(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    min_lr=1e-5,
    max_lr=1e0,
    early_stop_threshold=100,
)
print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()
net.hparams.learning_rate = res.suggestion()

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
trainer = pl.Trainer(
    max_epochs=5,
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    limit_train_batches=50,
    enable_checkpointing=True,
)


net = DeepAR.from_dataset(
    training,
    learning_rate=0.07,
    log_interval=10,
    log_val_interval=1,
    hidden_size=30,
    rnn_layers=2,
    optimizer="Adam",
    loss=MultivariateNormalDistributionLoss(rank=30),
)

trainer.fit(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
best_model_path = trainer.checkpoint_callback.best_model_path
best_model = DeepAR.load_from_checkpoint(best_model_path)
# best_model = net
predictions = best_model.predict(val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
MAE()(predictions.output, predictions.y)
raw_predictions = net.predict(
    val_dataloader, mode="raw", return_x=True, n_samples=100, trainer_kwargs=dict(accelerator="cpu")
)

series = validation.x_to_index(raw_predictions.x)["series"]
for idx in range(20):  # plot 10 examples
    best_model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
    plt.suptitle(f"Series: {series.iloc[idx]}")