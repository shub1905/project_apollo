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

start_time = time.time()

segments_timbre_size = 100

DATA_DIR = '/home/patanjali/courses/4772/project/MillionSongSubset/data/'

OUTPUT_FILE_DIR = '/home/patanjali/courses/4772/project/MillionSongSubset/data/'

#%%

def parse_data_files(folder=DATA_DIR):
    
    if not os.path.exists(DATA_DIR):
        print DATA_DIR, ' does not exist'
        raise

    file_path_gen = os.walk(DATA_DIR)
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
        artist_track_durations = pickle.load(open(OUTPUT_FILE_DIR+'artist_track_durations','r'))
        artist_names = pickle.load(open(OUTPUT_FILE_DIR+'artist_names','r'))
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
    
    pickle.dump(artist_track_durations, open(OUTPUT_FILE_DIR+'artist_track_durations','w'))
    pickle.dump(artist_names, open(OUTPUT_FILE_DIR+'artist_names','w'))
    
    return artist_names, artist_track_durations

def make_frame_wise(features, starts, min_duration, window_width):
    
    no_frames = int(min_duration/window_width)
    out = numpy.zeros((no_frames,features.shape[1]))
    inds = numpy.zeros((no_frames,), dtype='int')
    segment_counter = 0
    for i in xrange(no_frames):
        while segment_counter<(starts.shape[0]-1) and starts[segment_counter + 1] <= window_width*(i+1):
            segment_counter += 1
        inds[i] = segment_counter
    
    out = features.value[inds]
    out = out.reshape(out.shape[0]*out.shape[1])
    return out
    
def generate_data(min_duration=60, min_songs=10, window_width=.1, averaging='left'):
    
    artist_names, artist_track_durations = artist_mapping(from_stats_files=False)
    
    artist_track_counts = {key:len([t for t in artist_track_durations[key] if t >= min_duration]) 
                                        for key in artist_track_durations}
    artist_track_counts = {key:val for key, val in artist_track_counts.iteritems() 
                                        if val >= min_songs}

    counts = [val for (key, val) in artist_track_counts.iteritems()]
    print counts
    number_rows = sum(counts)
    artist_idxs = dict(zip(sorted(artist_track_counts.keys()), range(len(artist_track_counts))))
    
    no_frames = int(min_duration/window_width)
    no_columns = 1 + no_frames*24
    data = numpy.zeros((number_rows, no_columns))
    _files = parse_data_files()

    print 'Reading Data...'
    
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
        data[counter][1:(no_frames*12+1)] = make_frame_wise(timbres, starts, \
                                                            min_duration, window_width)
        data[counter][(no_frames*12+1):(no_frames*24+1)] = make_frame_wise(pitches, starts, \
                                                            min_duration, window_width)
        #data[counter][(min_segments*12+4)] = (starts[1:] - starts[:-1]).mean()
        #data[counter][(min_segments*12+2):(min_segments*12+14)] = timbres.value.mean(0)
        #data[counter][min_segments*12+1] = timbres.shape[0]
               
        counter += 1
        if counter % 10 == 0:
            print counter, time.time()-start_time
        f.close()
    
    data = data[:counter]
    
    rand_int = numpy.random.randint(0,100,counter)
    train_idx = (rand_int<60)
    rand_int = numpy.random.randint(0,100,sum(~train_idx))
    valid_idx = (rand_int<50)
    test_idx = (~valid_idx)
    
    numpy.save(OUTPUT_FILE_DIR + 'train', data[train_idx], allow_pickle=True)
    numpy.save(OUTPUT_FILE_DIR + 'valid', data[~train_idx][valid_idx], allow_pickle=True)
    numpy.save(OUTPUT_FILE_DIR + 'test', data[~train_idx][test_idx], allow_pickle=True)

if __name__ == '__main__':
    generate_data()
