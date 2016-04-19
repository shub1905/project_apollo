'''
File to generate mat files
Variable present in mat files
ArtistMapping(Dictionary): {artist_id: (echo_nest_id, artist_name)}
data(2D Array): {artist_id, MFCC}
'''


import h5py
import os
import scipy.io
import numpy
import pudb

segments_timbre_size = 100


def parse_data_files(folder='MillionSongSubset/data/'):
    pwd = os.getcwd()
    if not os.path.exists(pwd + '/' + folder):
        print pwd + '/' + folder, 'Does not exists'
        return

    MediaPath = pwd + '/' + folder
    file_path_gen = os.walk(MediaPath)
    for root, dirs, file_names in file_path_gen:
        for file_ in file_names:
            yield root + '/' + file_


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


hdf_song_object()
