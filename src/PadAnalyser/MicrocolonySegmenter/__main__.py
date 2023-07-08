print('here')


from PadAnalyser import FrameSets, OutputConfig
# import SegmentPads, SegmentPadsRunner


def segment_frame_set(frame_set: FrameSets.FrameSet, output_config: OutputConfig):

    print(frame_set, output_config)



    # SegmentPads.main(
    #     debug_output=args.debug_output,
    #     all_ignore_cache=args.all_ignore_cache,
    #     n_lim=args.n_lim,
    #     m_lim=args.m_lim,
    #     tiff_start_index=args.tiff_start_index,
    #     tiff_count=args.tiff_count,
    # )

# # to try to avoid mutliprocessing deadlock where no progess is made https://pythonspeed.com/articles/python-multiprocessing/
# from multiprocessing import set_start_method
# set_start_method("spawn")


# def main():
#     # Load input arguments from user.
#     parser = argparse.ArgumentParser(description='Analyse experiment with MAP pipeline.')
#     parser.add_argument('experiment', type=str, nargs='?', default=None, help='Experiment name.')
#     parser.add_argument('--debug_output', '-d', type=bool, nargs='?', const=True, default=False, help='Skip generating images.')
#     parser.add_argument('--all_ignore_cache', '-a', type=bool, nargs='?', const=True, default=False,  help='Do not skip files that have already been generated.')
#     parser.add_argument('--n_lim', '-n', type=int, nargs='?', default=None,  help='Limit the number of time-steps to analyze from each field of view.')
#     parser.add_argument('--m_lim', '-m', type=int, nargs='?', default=None,  help='Limit the number of fields of view to analyze.')
#     parser.add_argument('--tiff_start_index', '-ti', type=int, nargs='?', default=0, help='Generate tiff files starting at this index.')
#     parser.add_argument('--tiff_count', '-t', type=int, nargs='?', default=0, help='Generate this number of tiff files.')
#     args = parser.parse_args()

#     if args.experiment == None:
#         print('No experiment spesified, running all experiments.')
#         SegmentPadsRunner.main(
#             debug_output=args.debug_output,
#             all_ignore_cache=args.all_ignore_cache,
#             n_lim=args.n_lim,
#             m_lim=args.m_lim,
#             tiff_start_index=args.tiff_start_index,
#             tiff_count=args.tiff_count,
#         )
#         return

#     SegmentPads.main(
#         experiment=args.experiment, 
#         debug_output=args.debug_output,
#         all_ignore_cache=args.all_ignore_cache,
#         n_lim=args.n_lim,
#         m_lim=args.m_lim,
#         tiff_start_index=args.tiff_start_index,
#         tiff_count=args.tiff_count,
#     )

# if __name__ == '__main__':
#     main()