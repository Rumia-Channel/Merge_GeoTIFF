import argparse
from typing import Optional, Sequence

from .processing import DEFAULT_LANDSCAPE_RESOLUTIONS, main


def build_parser():
    parser = argparse.ArgumentParser(description="Merge GeoTIFF files and create a PNG image.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the input GeoTIFF files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the output files.")
    parser.add_argument("--sigma", type=float, default=0, help="Sigma value for the Gaussian blur. Default is 0 (no blur).")
    parser.add_argument("--output-graphs", action="store_true", help="Output a graph to visually recognise outliers. Can be very memory intensive.")
    parser.add_argument("--data-excluded", type=float, default=0, help="Remove outliers from the data by removing the upper %% and lower %%. Default is 0 (calculated based on all data).")
    parser.add_argument("--data-excluded_u", type=float, default=0, help="NEVER USE WITH --data-excluded. Remove outliers from the data by removing the upper %%. Default is 0.")
    parser.add_argument("--data-excluded_l", type=float, default=0, help="NEVER USE WITH --data-excluded. Remove outliers from the data by removing the lower %%. Default is 0.")
    parser.add_argument("--not-flat-earth", action="store_true", help="Produce height map data along the spherical shape of the earth.")
    parser.add_argument("--ue-landscape", action="store_true", help="Prepare data for Unreal Engine landscapes.")
    parser.add_argument("--small-units", action="store_true", help="ALWAYS USE WITH --ue-landscape. When binarising elevation data, binarise with elevation data for individual tiles.")
    parser.add_argument("--landscape-res", nargs="+", type=int, default=DEFAULT_LANDSCAPE_RESOLUTIONS, help='ALWAYS USE WITH --ue-landscape. Specifies the resolution at which the tiles are split into tiles. The default is all resolutions listed in the "Landscape Technical Guide".')
    return parser


def validate_args(parser, args):
    excluded_values = {
        "--data-excluded": args.data_excluded,
        "--data-excluded_u": args.data_excluded_u,
        "--data-excluded_l": args.data_excluded_l,
    }
    for option, value in excluded_values.items():
        if value < 0 or value > 100:
            parser.error(f"{option} must be between 0 and 100.")

    if args.data_excluded and (args.data_excluded_u or args.data_excluded_l):
        parser.error("--data-excluded cannot be used with --data-excluded_u or --data-excluded_l.")

    lower_excluded = args.data_excluded + args.data_excluded_l
    upper_excluded = args.data_excluded + args.data_excluded_u
    if lower_excluded + upper_excluded >= 100:
        parser.error("Combined upper and lower exclusion percentages must be less than 100.")

    if args.small_units and not args.ue_landscape:
        parser.error("--small-units requires --ue-landscape.")

    if args.landscape_res != DEFAULT_LANDSCAPE_RESOLUTIONS and not args.ue_landscape:
        parser.error("--landscape-res requires --ue-landscape.")

    invalid_resolutions = [res for res in args.landscape_res if res <= 0]
    if invalid_resolutions:
        parser.error("--landscape-res values must be positive integers.")


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return args


def cli(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    main(
        args.input_dir,
        args.output_dir,
        args.sigma,
        args.output_graphs,
        args.data_excluded,
        args.data_excluded_u,
        args.data_excluded_l,
        args.not_flat_earth,
        args.ue_landscape,
        args.small_units,
        args.landscape_res,
    )
