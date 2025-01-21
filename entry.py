import argparse
import os
from visualizer import generate_visualizer

parser = argparse.ArgumentParser(description="Generates the desired visualization for the provided music file.")
parser.add_argument(
    "-i", "--input",
    type=str,
    required=True,
    help="A required file path of the inputted music file (in .mp3 format)."
)
parser.add_argument(
    "-o", "--output",
    type=str,
    required=False,
    default="output.mp4",
    help=f"An optional argument to specify the path to store the music visualization (in .mp4 format)."
)
parser.add_argument(
    "-f", "--fps",
    type=int,
    required=False,
    default=30,
    help="The frames-per-second (fps) for the generated video."
)
parser.add_argument(
    "-p", "--prompt",
    type=str,
    required=False,
    help="A natural language description of the music visuals."
)

def get_ensured_absolute_path(input_path):
    if not os.path.isabs(input_path):
        return os.path.abspath(input_path)
    return input_path

def main():
    args = parser.parse_args()
    music_file = get_ensured_absolute_path(args.input)
    output_file = get_ensured_absolute_path(args.output)
    generate_visualizer(music_file, output_file, args.fps)


if __name__ == "__main__":
    main()