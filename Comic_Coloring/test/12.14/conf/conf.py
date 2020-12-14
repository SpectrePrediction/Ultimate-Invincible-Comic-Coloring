
# data reader 通用
image_txt_path = r"traintest.txt"
img_size = 224
read_data_cache = 0
batch_size = 20  # 总读取文件大小
reader_is_completion = True
reader_is_shuffle = True
reader_is_show_progress = True


# model 通用
model_args = {
    "star_channel": 64,
    "out_channel": 2,
    "mid_channel_scale": 0.5,
    "leaky_relu_alpha": 0.2,
    "groups": 32
}

# train 通用
learning_rate = 0.001

# train from fit
model_checkpoint_args = {
    "filepath": "my_model_test.h5",
    "monitor": 'val_loss',
    "verbose": 0,
    "save_best_only": True,
    "save_weights_only": True,
    "mode": 'auto',
    "period": 1
}

# train from fit
reduce_lr_on_plateau_args = {
    "monitor": 'val_loss',
    "factor": 0.7,
    "patience": 5,
    "mode": 'auto',
    "verbose": 0
}

# train from fit
fit_args = {
    "batch_size": 16,
    "epochs": 1000,
    "validation_split": 0.3
}

# train from static
pred_step_interval = 100
pred_image_path = "./image2/"
model_checkpoint_path = 'my_model_test.h5'

reduce_lr_static_args = {
    "patience": 5,
    "factor": 0.7
}