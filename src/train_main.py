def main():
    import argparse
    arg_parser = argparse.ArgumentParser(description='Train an image similarity vector embedding')
    arg_parser.add_argument("--epochs",metavar="N_epochs",type=int, nargs=1)
    
