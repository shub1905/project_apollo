import numpy


def read_data(raw_file, artist_map, artist_id_map):
    data = numpy.load(raw_file, mmap_mode='r')
    ArtistMapping = numpy.load(artist_map).item()
    ArtistIdMapping = numpy.load(artist_id_map).item()
    return data, ArtistMapping, ArtistIdMapping


def split_data(min_songs=25, max_artist=100, valid_split=.2, test_split=.2,
               raw_file='data/raw_songs_25.npy',
               artist_map='data/dict_songs_25.npy',
               artist_id_map='data/dictId_songs_25.npy'):

    data, ArtistMapping, ArtistIdMapping = read_data(raw_file, artist_map, artist_id_map)
    print 'loading done...'

    indices_tvt = [[], [], []]

    number_artists = 0
    for key in ArtistIdMapping:
        if number_artists == max_artist:
            break

        if len(ArtistIdMapping[key][1]) > min_songs:
            indices = ArtistIdMapping[key][1]
            valid_indx = int(valid_split * len(indices))
            indices_tvt[1] += map(lambda x: (x, number_artists), indices[:valid_indx])

            test_indx = valid_indx + max(int((len(indices) - 1) * test_split), 1)
            indices_tvt[2] += map(lambda x: (x, number_artists), indices[valid_indx:test_indx])

            indices_tvt[0] += map(lambda x: (x, number_artists), indices[test_indx:])
            number_artists += 1

    train = numpy.zeros((len(indices_tvt[0]), data.shape[1]))
    test = numpy.zeros((len(indices_tvt[2]), data.shape[1]))
    valid = numpy.zeros((len(indices_tvt[1]), data.shape[1]))

    data_arrays = [train, valid, test]
    data_arrays_name = ['train', 'valid', 'test']
    for i, arr in enumerate(data_arrays):
        for j, data_indx in enumerate(indices_tvt[i]):
            arr[j] = data[data_indx[0]]
            arr[j][0] = data_indx[1]
        numpy.save('data/' + data_arrays_name[i] + '_{}_{}'.format(min_songs, max_artist), arr)

# split_data(min_songs=100, max_artist=10000)
split_data()
