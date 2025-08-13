import pickle


def load_model():
    with open("../models/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def load_encoder():
    with open("../models/ohe.pkl", "rb") as f:
        one_hot_enc = pickle.load(f)
    return one_hot_enc
