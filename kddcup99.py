import argparse

def main():
    args = _init_args()
    data = _load_data(args)
    model = _init_model(args)
    _train_model(model, data, args)
    _save_model(model, args)

def _init_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="kddcup99.csv")
    return p.parse_args()

def _load_data(args):
    print("TODO: load data from %s" % args)
    return "XXX"

def _init_model(args):
    print("TODO: init model - but what?")
    return "YYY"

def _train_model(model, data, args):
    print("TODO: train ze model")

def _save_model(model, args):
    print("TODO: save model")

if __name__ == "__main__":
    main()
