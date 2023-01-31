def model_loader(model_specifications, num_classes):

    """This function returns a dictionary of loaded models.
    The model specification needs to be off the following form {\"name of model\" : (model_base, type, weights_path)}"""

    model_dict = {}

    for model_name in model_specifications.keys():

        base, simplicity, path = model_specifications[model_name]

        if base == 0:
            model = UNet(num_classes=num_classes, d_in=3,filters=[64, 128, 256, 512, 1024], simple=simplicity).to(device)
        elif base == 1:
            model = VGGUNet(num_classes, simplicity).to(device)
        elif base == 2:
            model = ResNetUNet(num_classes, simplicity).to(device)
        elif base == 3:
            model = EfficientNetUNet(num_classes, simplicity).to(device)
        else:
            print("Unknown model base selected")
            return -1
        
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)

        model_dict[model_name] = model
    
    return model_dict
