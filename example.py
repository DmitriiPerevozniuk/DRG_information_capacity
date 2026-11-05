"""One simulation, no plots and no network comparisons."""

import argparse
from pathlib import Path

import numpy as np

from model import LIF_with_CE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, help='Optional new .npz output file')
    args = parser.parse_args()
    if args.output and (args.output.suffix != '.npz' or args.output.exists()):
        parser.error('--output must be a new file with the .npz extension')
    time, voltage, agent, spikes, current, noise = LIF_with_CE(
        seed=args.seed, return_mask=True)
    print(f'{spikes.shape[1]} neurons, {len(time)} steps, {spikes.sum()} spikes')
    if args.output:
        with args.output.open('xb') as stream:
            np.savez_compressed(stream, time=time, voltage=voltage, agent=agent,
                                spikes=spikes, current=current, noise=noise,
                                seed=args.seed)
        print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
