import numpy

DATA = [numpy.load('/home/ubuntu/msds/data_'+char+'.npz')['data'] for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']

output = numpy.concatenate(DATA)

numpy.savez_compressed('/home/ubuntu/msds/data_all.npz', output)

