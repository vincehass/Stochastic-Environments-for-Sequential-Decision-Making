import os
import gzip
import pickle
import numpy as np

class Logger:
    def __init__(self, args):
        self.args = args
        self.data = {}

    def add_scalar(self, name, value, step):
        if name not in self.data:
            self.data[name] = []
        self.data[name].append((step, value))

    def add_object(self, name, obj):
        self.data[name] = obj

    def save(self, save_path, args):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        pickle.dump({'logged_data': self.data, 'args': self.args}, gzip.open(save_path, 'wb'))

    def load(self, load_path):
        with gzip.open(load_path, 'rb') as f:
            loaded_data = pickle.load(f)
        self.data = loaded_data['logged_data']
        self.args = loaded_data['args']

    def get_scalar(self, name):
        return np.array(self.data[name])

    def get_object(self, name):
        return self.data[name]

def get_logger(args):
    return Logger(args)
