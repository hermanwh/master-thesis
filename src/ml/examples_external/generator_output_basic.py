# multivariate one step problem
from numpy import array
from numpy import hstack
from numpy import insert
from keras.preprocessing.sequence import TimeseriesGenerator
# define dataset
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105])
out_seq = array([25, 45, 65, 85, 105, 125, 145, 165, 185, 205])
# reshape series
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2))
# shift the target sample by one step
out_seq = insert(out_seq, 0, 0)
# define generator
n_input = 1
generator = TimeseriesGenerator(dataset, out_seq, length=n_input, batch_size=1)
# print each sample
for i in range(len(generator)):
	x, y = generator[i]
	print('%s => %s' % (x, y))