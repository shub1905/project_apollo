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
import numpy

segments_timbre_size = 100

DATA_DIR = '/home/patanjali/courses/4772/project/MillionSongSubset/data/'

OUTPUT_FILE_DIR = '/home/patanjali/courses/4772/project/MillionSongSubset/data/'

sample_file = 'A/A/A/TRAAAAW128F429D538.h5'

f = h5py.File(DATA_DIR + sample_file, 'r')
f['analysis']

#%%

def parse_data_files(folder=DATA_DIR):
    
    if not os.path.exists(DATA_DIR):
        print DATA_DIR, ' does not exist'
        raise

    file_path_gen = os.walk(DATA_DIR)
    for root, dirs, file_names in file_path_gen:
        for file_ in file_names:
            yield root + '/' + file_

def artist_mapping(min_songs=4):
    # TODO: use summary file in additional files instead of parsing whole dataset
    '''
    reads hdf files and stores their attributes
    songs with timbre < min_timbre are ignored
    every artist must have atleast min_songs in the dataset
    '''
    print 'Generating Mapping...'
    _files = parse_data_files()
    artist_names = {}
    artist_track_counts = defaultdict(int)

    for file_name in _files:
        # print file_name
        if not file_name.endswith('.h5'):
            continue
        f = h5py.File(file_name, 'r')

        f_meta = f['metadata']['songs']
        f_artist_id_indx = f_meta.dtype.names.index('artist_id')
        f_artist_name_indx = f_meta.dtype.names.index('artist_name')
        f_artist_id = f_meta[0][f_artist_id_indx]
        f_artist_name = f_meta[0][f_artist_name_indx]

        artist_names[f_artist_id] = f_artist_name
        artist_track_counts[f_artist_id] += 1

    artist_track_counts = {key:artist_track_counts[key] for key in artist_track_counts 
                                            if artist_track_counts[key] >= min_songs}
    
    artist_names = {key:artist_names[key] for key in artist_names
                                                        if key in artist_track_counts}
    
    return artist_names, artist_track_counts

"""

def split_data(min_timbre=100, timbre_width=12, min_songs=4, data_file='mfcc'):
    ArtistMapping, ArtistIdMapping, data = generate_data(min_timbre=min_timbre, 
                                                         timbre_width=timbre_width, 
                                                         min_songs=min_songs)
    train = numpy.zeros((0, min_timbre * timbre_width + 1))
    validation = numpy.zeros((0, min_timbre * timbre_width + 1))
    test = numpy.zeros((0, min_timbre * timbre_width + 1))

    print 'Splitting data...'
    for artist_id in ArtistIdMapping:
        indices = ArtistIdMapping[artist_id][1]

        valid_data = data[indices[0]].reshape(1, -1)
        validation = numpy.vstack((validation, valid_data))

        test_indx = 1 + max(int((len(indices) - 1) * .3), 1)

        for i in range(1, test_indx):
            test_data = data[indices[i]].reshape(1, -1)
            test = numpy.vstack((test, test_data))

        for i in range(test_indx, len(indices)):
            train_data = data[indices[i]].reshape(1, -1)
            train = numpy.vstack((train, train_data))

    print 'Saving Data...'

    numpy.save('data/' + data_file + '_test', test, allow_pickle=True)
    numpy.save('data/' + data_file + '_train', train, allow_pickle=True)
    numpy.save('data/' + data_file + '_valid', validation, allow_pickle=True)
    return ArtistMapping, ArtistIdMapping, train, validation, test

"""

def generate_data(min_segments=60, min_songs=10):
    
    #ArtistMapping, ArtistIdMapping = artist_mapping(min_songs=min_songs)
    artist_names, artist_track_counts = artist_mapping(min_songs=min_songs)
    number_rows = sum([val for (key, val) in artist_track_counts.iteritems()])
    artist_idxs = dict(zip(sorted(artist_names.keys()), range(len(artist_names))))

    number_columns = 1 + min_segments * 12 + 14
    data = numpy.zeros((number_rows, number_columns))
    _files = parse_data_files()
    counter = 0

    print 'Reading Data...'
    
    ### Adding the additional "dumb" meta-features - number of segments, averages of
    ### timbres and average segment length
    
    for file_name in _files:
        if not file_name.endswith('.h5'):
            continue

        f = h5py.File(file_name, 'r')

        _meta = f['metadata']['songs']
        _analysis = f['analysis']
        
        starts = _analysis['segments_start']
        timbres = _analysis['segments_timbre']

        _artist_id_indx = _meta.dtype.names.index('artist_id')
        _artist_id = _meta[0][_artist_id_indx]
        artist_idx = artist_idxs[_artist_id]

        if timbres.shape[0] < min_segments:
            # skip adding song features is timbre shape < min_timbre
            # or song_count is less than min_songs
            continue
        
        segments = timbres[:min_segments].reshape(min_segments*12)
        data[counter][(min_segments*12+4)] = (starts[1:] - starts[:-1]).mean()
        data[counter][(min_segments*12+2):(min_segments*12+14)] = timbres.value.mean(0)
        data[counter][min_segments*12+1] = timbres.shape[0]
        data[counter][1:(min_segments*12+1)] = segments
        data[counter][0] = artist_idx
        counter += 1
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

"""
def hdf_song_object(min_timbre=100, timbre_width=12):
    '''
    reads hdf files and stores their attributes
    songs with timbre < min_timbre are ignored
    '''

    gen = parse_data_files()
    ArtistMapping = {}
    artist_map_id_echo = {}
    data = numpy.zeros((0, min_timbre * timbre_width + 1))
    artist_count = 0

    for file_name in gen:
        f = h5py.File(file_name, 'r')
        print file_name

        f_meta = f['metadata']['songs']
        f_analysis = f['analysis']['segments_timbre']

        f_artist_id_indx = f_meta.dtype.names.index('artist_id')
        f_artist_name_indx = f_meta.dtype.names.index('artist_name')
        f_artist_id = f_meta[0][f_artist_id_indx]
        f_artist_name = f_meta[0][f_artist_name_indx]

        if f_analysis.shape[0] < min_timbre:
            # skip adding song features is timbre shape < min_timbre
            continue

        current_artist_count = ArtistMapping.get(f_artist_id, (artist_count, 'foo'))[0]
        ArtistMapping[f_artist_id] = (current_artist_count, f_artist_name)
        artist_map_id_echo[str(current_artist_count)] = (f_artist_id, f_artist_name.strip())
        artist_count = artist_count + 1

        segments = f_analysis[:min_timbre].reshape(1, min_timbre * timbre_width)
        segments = numpy.insert(segments, 0, current_artist_count, axis=1)
        data = numpy.vstack((data, segments))
        f.close()

    matlab_file_mfcc = open('data/dataMFCC.mat', 'w')
    scipy.io.savemat(matlab_file_mfcc, {'data': data})

    matlab_file_artist = open('data/dataArtist.mat', 'w')
    scipy.io.savemat(matlab_file_artist, artist_map_id_echo)
"""