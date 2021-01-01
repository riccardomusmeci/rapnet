import codecs
import os
import collections
import numpy as np

sequence_length = 5
batch_size = 32

data_dir = "data/kanye"
with codecs.open(os.path.join(data_dir, "input.txt"), "r", encoding="utf-8") as file:
    data = file.read()

count_pairs = sorted(collections.Counter(data).items(), key=lambda x: -x[1])
pointer = 0
chars, a = zip(*count_pairs)

# print(chars)
vocabulary_size = len(chars)
vocabulary = dict(zip(chars, range(len(chars))))
print(vocabulary)
quit()
tensor = np.array(list(map(vocabulary.get, data)))
# print(tensor.shape)

batches_size = int(tensor.size / (batch_size * sequence_length))
# print(batches_size)

print(tensor.shape)
tensor = tensor[:batches_size * batch_size * sequence_length]
print(tensor.shape)

inputs = tensor
targets = np.copy(tensor)

targets[:-1] = inputs[1:]
targets[-1] = inputs[0]
input_batches = np.split(inputs.reshape(batch_size, -1), batches_size, 1)
target_batches = np.split(targets.reshape(batch_size, -1), batches_size, 1)
print(input_batches[0][0])
print(target_batches[0][0])

# class DataProvider:

#     def __init__(self, data_dir, batch_size, sequence_length):
#         self.batch_size = batch_size
#         self.sequence_length = sequence_length
#         with codecs.open(os.path.join(data_dir, "input.txt"), "r", encoding="utf-8") as file:
#             data = file.read()
#         count_pairs = sorted(collections.Counter(data).items(), key=lambda x: -x[1])
#         self.pointer = 0
#         self.chars, _ = zip(*count_pairs)
#         self.vocabulary_size = len(self.chars)
#         self.vocabulary = dict(zip(self.chars, range(len(self.chars))))
#         self.tensor = np.array(list(map(self.vocabulary.get, data)))
#         self.batches_size = int(self.tensor.size / (self.batch_size * self.sequence_length))
#         if self.batches_size == 0:
#             assert False, "Unable to generate batches. Reduce batch_size or sequence_length."

#         self.tensor = self.tensor[:self.batches_size * self.batch_size * self.sequence_length]
#         inputs = self.tensor
#         targets = np.copy(self.tensor)

#         targets[:-1] = inputs[1:]
#         targets[-1] = inputs[0]
#         self.input_batches = np.split(inputs.reshape(self.batch_size, -1), self.batches_size, 1)
#         self.target_batches = np.split(targets.reshape(self.batch_size, -1), self.batches_size, 1)
#         print "Tensor size: " + str(self.tensor.size)
#         print "Batch size: " + str(self.batch_size)
#         print "Sequence length: " + str(self.sequence_length)
#         print "Batches size: " + str(self.batches_size)
#         print ""

#     def next_batch(self):
#         inputs = self.input_batches[self.pointer]
#         targets = self.target_batches[self.pointer]
#         self.pointer += 1
#         return inputs, targets

#     def reset_batch_pointer(self):
#         self.pointer = 0
