import codecs
import os
import collections
import numpy as np
import logging

class DataLoader:

    def __init__(self, data_dir: str, batch_size: int, sequence_length: int):
        """__init__ Inits the DataLoader

        Args:
            data_dir (str): path to the dataset (txt file with lyrics)
            batch_size (int): size of the batch to produce
            sequence_length (int): length of each element in the batch
        """

        self.data = None
        # Loads the dataset into self.data
        self.__load_dataset(data_dir=data_dir)
        
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        
        self.__prepare_data()

    def __load_dataset(self, data_dir: str):
        """__load_dataset checks if dataset exists and loads the data

        Args:
            data_dir (str): path to the dataset (txt file with lyrics)
        """

        try:
            with codecs.open(data_dir, "r", encoding="utf-8") as f:
                self.data = f.read()
            f.close()
        except FileExistsError:
            logging.error("No dataset found at "+ data_dir)
            exit(1)

    def __prepare_data(self):
        """__prepare_data prepares the data for the DL model
        """
        # Creating the pairs ("char": counter) and sorting this iterable in descent order based on the counter
        count_pairs = sorted(collections.Counter(self.data),items(), key=lambda x: -x[1])
        # Getting the chars from the count_pairs
        self.chars, _ = zip(*count_pairs)
        self.vocabulary_size = len(self.chars)
        # Creating the vocubulary as a dict { char: coding_val } (e.g. { 'a': 0, 'b': 1, ..})
        self.vocabulary = dict(zip(self.chars, range(len(self.chars))))

        # Creating the tensors by applying the "get" function to the vocabulary (that gives me the coding_val of a char )
        # and substituing the value in self.data
        self.tensor_data = np.array(list(map(self.vocabulary.get, self.data)))

        # Number of batches to produce
        self.batches_size = int(self.tensor_data.size / (self.batch_size * self.sequence_length))

        if self.batches_size == 0:
            assert False, "Unable to generate batches. Reduce batch_size or sequence_length."
        
        # Getting the tensor data for the batches
        self.tensor_data = self.tensor_data[:self.batches_size * self.batch_size * self.sequence_length]

        # Creating the inputs and the targets. Each input consists of a word and the target is the same
        #  word and its last char is the next char of the initial word
        # e.g. inpui: (K a n y) -> target: ( a n y e)
        inputs = self.tensor_data
        targets = np.copy(self.tensor_data)
        targets[:-1] = inputs[1:]
        targets[-1] = inputs[0]

        # Creating the input and target batches
        self.input_batches = np.split(inputs.reshape(self.batch_size, -1), self.batches_size, 1)
        self.target_batches = np.split(targets.reshape(self.batch_size, -1), self.batches_size, 1)
    
    def data_generator(self) -> tuple:
        """data_generator __data_generator data generator for each batch

        Returns:
            tuple: input batch, target batch

        Yields:
            Iterator[tuple]: generator for input and target batch
        """
        
        for input_data, target_data in zip(self.input_batches, self.target_batches):
            yield input_data, target_data






