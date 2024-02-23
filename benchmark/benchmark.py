import argparse
import torch
from datetime import datetime


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Ensure that you have CUDA installed and your GPU supports it.")

    # Run the testing script and measure time

    # Calculate the time per frame

    # Write the results to a text file

    gpu_name = torch.cuda.get_device_name(0)

    date_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results = []

    write_results(results, date_string, gpu_name)


def write_results():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--output", default="benchmark_results.txt", help="")
    args = parser.parse_args()

    main(args.input, args.output)
