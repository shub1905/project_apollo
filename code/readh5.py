'''
File to generate mat files
Variable present in mat files
ArtistMapping(Dictionary): {artist_id: (echo_nest_id, artist_name)}
data(2D Array): {artist_id, MFCC}
'''


import h5py
import os
import sys
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


def artist_mapping(min_songs=4):
    # TODO: use summary file in additional files instead of parsing whole dataset
    '''
    reads hdf files and stores their attributes
    songs with timbre < min_timbre are ignored
    every artist must have atleast min_songs in the dataset
    '''
    print 'Generating Mapping...'
    gen = parse_data_files()
    ArtistMapping = {}

    for file_name in gen:
        # print file_name
        if not file_name.endswith('.h5'):
            continue
        f = h5py.File(file_name, 'r')

        f_meta = f['metadata']['songs']
        f_artist_id_indx = f_meta.dtype.names.index('artist_id')
        f_artist_name_indx = f_meta.dtype.names.index('artist_name')
        f_artist_id = f_meta[0][f_artist_id_indx]
        f_artist_name = f_meta[0][f_artist_name_indx]

        temp = ArtistMapping.get(f_artist_id, [0, 'foo'])[0]
        ArtistMapping[f_artist_id] = (temp + 1, f_artist_name)

    removal_arts = []

    for art in ArtistMapping:
        if ArtistMapping[art][0] < min_songs:
            removal_arts.append(art)

    for art in removal_arts:
        del(ArtistMapping[art])

    current_artist = 0
    ArtistIdMapping = {}

    for art in ArtistMapping:
        temp = ArtistMapping[art]
        ArtistIdMapping[str(current_artist)] = (art, [])
        ArtistMapping[art] = (temp[0], temp[1], current_artist)
        current_artist += 1

    return ArtistMapping, ArtistIdMapping


def generate_data(min_timbre=100, timbre_width=12, min_songs=4):
    ArtistMapping, ArtistIdMapping = artist_mapping(min_songs=min_songs)
    number_rows = sum(map(lambda x: x[0], ArtistMapping.values()))

    data = numpy.zeros((number_rows, min_timbre * timbre_width + 1))
    gen = parse_data_files()
    counter = 0

    print 'Reading Data...'
    for file_name in gen:
        if not file_name.endswith('.h5'):
            continue

        f = h5py.File(file_name, 'r')

        f_meta = f['metadata']['songs']
        f_analysis = f['analysis']['segments_timbre']

        f_artist_echo_id_indx = f_meta.dtype.names.index('artist_id')
        f_artist_echo_id = f_meta[0][f_artist_echo_id_indx]
        artist_id = ArtistMapping.get(f_artist_echo_id, (-1, -1, -1))[2]

        if f_analysis.shape[0] < min_timbre or artist_id == -1:
            # skip adding song features is timbre shape < min_timbre
            # or song_count is less than min_songs
            continue

        # print file_name
        ArtistIdMapping[str(artist_id)][1].append(counter)
        segments = f_analysis[:min_timbre].reshape(min_timbre * timbre_width)
        data[counter][1:] = segments
        data[counter][0] = artist_id
        counter += 1
        f.close()

    return ArtistMapping, ArtistIdMapping, data


def split_data(min_timbre=100, timbre_width=12, min_songs=4, data_file='mfcc'):
    ArtistMapping, ArtistIdMapping, data = generate_data(
        min_timbre=min_timbre, timbre_width=timbre_width, min_songs=min_songs)
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

    numpy.save('data/' + data_file + '_songs_{}'.format(min_songs) + '_test', test, allow_pickle=True)
    numpy.save('data/' + data_file + '_songs_{}'.format(min_songs) + '_train', train, allow_pickle=True)
    numpy.save('data/' + data_file + '_songs_{}'.format(min_songs) + '_valid', validation, allow_pickle=True)
    numpy.save('data/' + data_file + '_dict_songs_{}'.format(min_songs), ArtistMapping, allow_pickle=True)
    numpy.save('data/' + data_file + '_dictId_songs_{}'.format(min_songs), ArtistIdMapping, allow_pickle=True)
    return ArtistMapping, ArtistIdMapping, train, validation, test


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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: python readh5.py min_songs_per_artist'
        sys.exit()

    min_sngs = int(sys.argv[1])
    split_data(min_songs=min_sngs)