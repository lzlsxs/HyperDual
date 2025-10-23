from models.our.our import Our
def get_model(model_name, args):
    name = model_name.lower()
    if name == "our":
        return Our(args)
    else:
        assert 0
