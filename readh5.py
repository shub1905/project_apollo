import h5py
import os


def parse_data_files(folder='MillionSongSubset/data/'):
    pwd = os.getcwd()
    if not os.path.exists(pwd + '/' + folder):
        print pwd + '/' + folder, 'Does not exists'
        return

    MediaPath = pwd + '/' + folder
    file_path_gen = os.walk(MediaPath)
    for root, dirs, file_names in file_path_gen:
        for file_ in file_names:
            yield root+'/'+file_


def hdf_song_object():
	gen = parse_data_files()
	names = open('SongNames.txt','a')
	for file_name in gen:
		f = h5py.File(file_name, 'r')
		f_meta = f['metadata']['songs']
		f_title_indx = f_meta.dtype.names.index('title')
		names.write(f_meta[0][f_title_indx]+'\n')
	names.close()

hdf_song_object()