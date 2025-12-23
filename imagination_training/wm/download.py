from modelscope.hub.file_download import model_file_download

local_path = model_file_download(
    model_id='AI-ModelScope/mineworld',
    cache_dir="./imagination_training/mineworld",
    file_path='checkpoints/700M_16f.ckpt',  # relative path as shown in the web UI
    revision='master',
)
print(local_path)