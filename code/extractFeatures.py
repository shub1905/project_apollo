'''
File to generate mat files
Variable present in mat files
ArtistMapping(Dictionary): {artist_id: (echo_nest_id, artist_name)}
data(2D Array): {artist_id, MFCC}
'''

#%%

from collections import defaultdict
import h5py
import os
import pickle
import time

import numpy
import gc

start_time = time.time()

segments_timbre_size = 100

#DATA_DIR = '/home/patanjali/courses/4772/project/MillionSongSubset/data/'
#OUTPUT_FILE_DIR = '/home/patanjali/courses/4772/project/MillionSongSubset/data/'
#META_FILE_DIR = OUTPUT_FILE_DIR#'/home/patanjali/courses/4772/project/project_apollo/data/'
import sys
#print sys.argv[1], sys.argv[2]

DATA_DIR = '/home/ubuntu/msds/data/'
# + sys.argv[1] + '/' + sys.argv[2] + '/'
OUTPUT_FILE_DIR = '/home/ubuntu/msds/'
META_FILE_DIR = '/home/ubuntu/project_apollo/data/'

#%%

def parse_data_files(folder=DATA_DIR):
    
    if not os.path.exists(folder):
        print folder, ' does not exist'
        raise

    file_path_gen = os.walk(folder)
    for root, dirs, file_names in file_path_gen:
        for file_ in file_names:
            yield root + '/' + file_

def artist_mapping(from_stats_files=False):
    # TODO: use summary file in additional files instead of parsing whole dataset
    '''
    reads hdf files and stores their attributes
    songs with timbre < min_timbre are ignored
    every artist must have atleast min_songs in the dataset
    '''
    if from_stats_files:
        artist_track_durations = pickle.load(open(META_FILE_DIR+'artist_track_durations','r'))
        artist_names = pickle.load(open(META_FILE_DIR+'artist_names','r'))
        return artist_names, artist_track_durations
    
    print 'Generating Mapping...'
    _files = parse_data_files()
    artist_names = {}
    artist_track_counts = defaultdict(int)
    artist_track_durations = defaultdict(list)
    counter = 0
    for file_name in _files:
        counter += 1
        if counter % 100 == 0:
            print counter, time.time()-start_time
        # print file_name
        if not file_name.endswith('.h5'):
            continue
        
        f = h5py.File(file_name, 'r')

        f_meta = f['metadata']['songs']
        f_artist_id_indx = f_meta.dtype.names.index('artist_id')
        f_artist_name_indx = f_meta.dtype.names.index('artist_name')
        f_artist_id = f_meta[0][f_artist_id_indx]
        f_artist_name = f_meta[0][f_artist_name_indx]
        if f['analysis']['segments_start'].shape[0]==0:
            continue        
        
        artist_names[f_artist_id] = f_artist_name
        artist_track_counts[f_artist_id] += 1
        if len(artist_track_durations[f_artist_id]) < artist_track_counts[f_artist_id]:
            artist_track_durations[f_artist_id].extend([0]*10)
        
        artist_track_durations[f_artist_id][artist_track_counts[f_artist_id]-1] = \
                                            f['analysis']['segments_start'][-1]
        
        f.close()
    
    artist_track_durations = {key:value[:artist_track_counts[key]]
                                    for key, value in artist_track_durations.iteritems()}
    
    pickle.dump(artist_track_durations, open(META_FILE_DIR+'artist_track_durations','w'))
    pickle.dump(artist_names, open(META_FILE_DIR+'artist_names','w'))
    
    return artist_names, artist_track_durations

def make_frame_indices(starts, min_duration, window_width):
    no_frames = int(min_duration/window_width)
    frame_indices = numpy.zeros((no_frames,), dtype='int')
    segment_counter = 0
    
    '''
    for i in xrange(no_frames):
        while segment_counter<(starts.shape[0]-1) and starts[segment_counter + 1] <= window_width*(i+1):
            segment_counter += 1
        inds[i] = segment_counter
    '''
    #Alternate formulation of previous
    segment_counter_end = int(starts[0]/window_width)
    frame_indices[segment_counter:segment_counter_end] = 0
    segment_counter = segment_counter_end
    for i in xrange(starts.shape[0]-1):
        _temp = int(starts[i+1]/window_width)
        segment_counter_end = min([no_frames, _temp])
        frame_indices[segment_counter:segment_counter_end] = i+1
        segment_counter = segment_counter_end
        if _temp >= no_frames:
            break
    
    return frame_indices
    

def make_frame_wise(features, frame_indices):

    out = features.value[frame_indices].copy()
    out = out.reshape(out.shape[0]*out.shape[1])
    return out
    
def generate_data(min_duration=60, min_songs=10, window_width=.1, averaging='left', 
                  folder=DATA_DIR, artist_track_durations=artist_mapping(from_stats_files=True)[1], 
                    suffix='', memmap=False):
    
    
    artist_track_counts = {key:len([t for t in artist_track_durations[key] if t >= min_duration]) 
                                        for key in artist_track_durations}
    artist_track_counts = {key:val for key, val in artist_track_counts.iteritems() 
                                        if val >= min_songs}

    counts = [val for (key, val) in artist_track_counts.iteritems()]
    #print counts
    number_rows = sum(counts)
    artist_idxs = dict(zip(sorted(artist_track_counts.keys()), range(len(artist_track_counts))))
    #del artist_names
    #del artist_track_durations
    del artist_track_counts
    gc.collect()
    
    no_frames = 0#int(min_duration/window_width)
    no_columns = 1 + no_frames*24 + 180
    '''
    for _i in xrange(100):
        if not os.path.isfile(OUTPUT_FILE_DIR+'temp_'+str(_i)):
            break
    data = numpy.memmap(OUTPUT_FILE_DIR+'temp_'+str(_i), dtype='float32', \
                            mode='w+', shape=(number_rows, no_columns))
    '''
    if memmap:
        data = numpy.memmap(OUTPUT_FILE_DIR+'temp_'+suffix, dtype='float32', \
                            mode='w+', shape=(number_rows, no_columns))
    else:
        data = numpy.zeros((number_rows, no_columns), dtype='float32')
    _files = parse_data_files(folder)

    print 'Reading Data...'
    print number_rows, no_columns, len(artist_idxs)
    ### Adding the additional "dumb" meta-features - number of segments, averages of
    ### timbres and average segment length
    counter = 0
    for file_name in _files:
        if not file_name.endswith('.h5'):
            continue

        f = h5py.File(file_name, 'r')

        _meta = f['metadata']['songs']
        _analysis = f['analysis']
        
        starts = _analysis['segments_start']
        timbres = _analysis['segments_timbre']
        pitches = _analysis['segments_pitches']
        
        _artist_id_indx = _meta.dtype.names.index('artist_id')
        _artist_id = _meta[0][_artist_id_indx]
        
        if _artist_id not in artist_idxs or min_duration > starts[-1]:
            # skip adding song features is <DELETED>
            # or song_count is less than min_songs
            f.close()
            continue
        
        artist_idx = artist_idxs[_artist_id]
        data[counter][0] = artist_idx
        
        #frame_indices = make_frame_indices(starts, min_duration, window_width)
        #data[counter][1:(no_frames*12+1)] = make_frame_wise(timbres, frame_indices)
        #data[counter][(no_frames*12+1):(no_frames*24+1)] = make_frame_wise(pitches, frame_indices)
        
        data[counter][(no_frames*24+1):(no_frames*24+13)] = numpy.mean(timbres,0)
        data[counter][(no_frames*24+13):(no_frames*24+91)] = \
                            numpy.cov(timbres, rowvar=False)[numpy.triu_indices(12)]
        data[counter][(no_frames*24+91):(no_frames*24+103)] = numpy.mean(pitches,0)
        data[counter][(no_frames*24+103):(no_frames*24+181)] =\
                            numpy.cov(pitches, rowvar=False)[numpy.triu_indices(12)]
        #data[counter][(min_segments*12+4)] = (starts[1:] - starts[:-1]).mean()
        #data[counter][(min_segments*12+2):(min_segments*12+14)] = timbres.value.mean(0)
        #data[counter][min_segments*12+1] = timbres.shape[0]
               
        counter += 1
        if counter % 1000 == 0:
            print counter, time.time()-start_time
        f.close()
    
    '''
    data = data[:counter]
    
    rand_int = numpy.random.randint(0,100,counter)
    train_idx = (rand_int<60)
    rand_int = numpy.random.randint(0,100,sum(~train_idx))
    valid_idx = (rand_int<50)
    test_idx = (~valid_idx)
    
    numpy.save(OUTPUT_FILE_DIR + 'train', data[train_idx], allow_pickle=True)
    numpy.save(OUTPUT_FILE_DIR + 'valid', data[~train_idx][valid_idx], allow_pickle=True)
    numpy.save(OUTPUT_FILE_DIR + 'test', data[~train_idx][test_idx], allow_pickle=True)
    '''
    if memmap:
        output = numpy.memmap(OUTPUT_FILE_DIR+'features_'+suffix, \
                                       dtype='float32', mode='w+', shape=(counter, no_columns))
        data.flush()
        output.flush()
    else:
        data = data[:counter, :]
        numpy.savez_compressed(OUTPUT_FILE_DIR+'data_'+suffix+'.npz', data=data)
        
if __name__ == '__main__':

    artist_names, artist_track_durations = artist_mapping(from_stats_files=True)
    '''
    for fldr1 in 'JKLMNOPQRSTUVWXYZ':
        for fldr2 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            print DATA_DIR+fldr1+'/'+fldr2+'/'
            generate_data(folder=DATA_DIR+fldr1+'/'+fldr2+'/', artist_track_durations=artist_track_durations, suffix=fldr1+'_'+fldr2)
    '''
    for fldr1 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        print fldr1
        generate_data(folder=DATA_DIR+fldr1+'/', artist_track_durations=artist_track_durations, 
                          suffix=fldr1)
    