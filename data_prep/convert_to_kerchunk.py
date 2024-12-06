from kerchunk.hdf import SingleHdf5ToZarr
import fsspec
import ujson


def generate_json_from_nc(u,fs, fs_out, so, json_dir: str):
    with fs.open(u, **so) as infile:
        h5chunks = SingleHdf5ToZarr(infile, u, inline_threshold=300)

        file_split = u.split('/') # seperate file path to create a unique name for each json 
        model = file_split[1].split('_')[0]
        date_string = file_split[-1].split('_')[3]
        outf = f'{json_dir}{model}_{date_string}.json'
        with fs_out.open(outf, 'wb') as f:
            f.write(ujson.dumps(h5chunks.translate()).encode());