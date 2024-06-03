def get_model_name(model_checkpoint: str):
    return "_".join(model_checkpoint.split("/")[-1].split("-")[:-1])
