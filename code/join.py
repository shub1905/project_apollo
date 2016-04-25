import numpy

DIR = '/home/patanjali/courses/4772/project/project_apollo/data/'

#%%

DATA = [numpy.load(DIR+'data_'+char+'.npz')['data'] for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']

output = numpy.concatenate(DATA)

numpy.savez_compressed(DIR+'data_all.npz', output)

rand_int = numpy.random.randint(0,100,output.shape[0])
train_idx = (rand_int<60)
rand_int = numpy.random.randint(0,100,sum(~train_idx))
valid_idx = (rand_int<50)
test_idx = (~valid_idx)

#%%

numpy.savez_compressed(DIR + 'train', output[train_idx])
numpy.savez_compressed(DIR + 'valid', output[~train_idx][valid_idx])
numpy.savez_compressed(DIR + 'test', output[~train_idx][test_idx])