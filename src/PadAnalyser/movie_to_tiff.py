import os
from PIL import Image
from TemikaXML.Movie import Movie
import MKUtils
import argparse


def to_image(frame, dir, filename):

    file_path = os.path.join(dir, filename)
    try:
        # if normalize:
        #    frame = frame[:, :, np.newaxis]
        #    frame = (255*(frame - np.min(frame)) /
        #             np.ptp(frame)).astype(int)
        image = Image.fromarray(frame)
        image.save(file_path)
        return 1

    except Exception as e:
        print(f"Could not write image {file_path}.", e)
    return 0

def main(movie, interval, n_lim, start_index, output_directory = None):

    # 1. Load movie file
    try:
        m = Movie(movie)
    except Exception as e:
        print('Could not import movie file.', e)
        return

    experiment_name = os.path.basename(movie).split('.')[0]
    folder = os.path.dirname(movie)

    output_dir = output_directory or os.path.join(folder, 'images')
    MKUtils.generate_directory(output_dir)
    print(output_dir)

    # 2. Generate index sets for each imaging field of view.

    interval = interval
    n_lim = n_lim
    start_index = start_index
    end_index = start_index + interval*n_lim if n_lim else len(m)
    end_index = min(len(m), end_index)

    indices = range(start_index, end_index, interval)
    successes = 0

    for j, i in enumerate(indices):
        successes += to_image(
            frame=m[i], 
            dir=output_dir, 
            filename=f'{experiment_name}_i{start_index+interval*j}.tiff'
        )
    print(f'{successes} images generated.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract information from .movie files.')
    parser.add_argument('movie', type=str,
                        help='Filename for source .movie file.')
    parser.add_argument('--start_index', '-s', type=int, nargs='?', default=0, 
                        help='Spesify first infex to be converted.')
    parser.add_argument('--interval', '-i', type=int, nargs='?', default=1, 
                        help='Interval between converted frames.')
    parser.add_argument('--n_lim', '-n', type=int, nargs='?', default=None, 
                        help='Limit the number of time-steps to analyze from each field of view.')
    args = parser.parse_args()

    print(f'Generateing images from file {args.movie}', end=': ')
    print(f'starting at index {args.start_index} with interval {args.interval}', end=', ')
    print(f'{"all timesteps" if args.n_lim == None else f"limited to {args.n_lim} timesteps"}')

    main(movie=args.movie, interval=args.interval, n_lim=args.n_lim, start_index=args.start_index)