def create_filename(directory: str, seed: int, params: list = []):
    name = str(seed)
    name += ['_'+str(param).replace('.', '') for param in params]
    name += '.pth'
    return directory + name
