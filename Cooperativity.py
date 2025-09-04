import pathlib
import numpy as np
import WHAM
import glob
import argparse
def main():

    parser = argparse.ArgumentParser(description='Intake the working directory to generate, execute and analyze Gromacs files')
    parser.add_argument('directory', help='Path to the directory')
    args = parser.parse_args()

    # Use the directory argument
    directory_path = args.directory
    return directory_path

if __name__ == '__main__':
    wrkdir = main()
def cooperativity_check(wrkdir):
    xvg = glob.glob(f'{wrkdir}/*.xvg')
    for file in xvg:
        idx = len(xvg)
        while idx > 0:
            lines = np.loadtxt(file, comments=['@', '#'], usecols=[1])
            idx = len(lines)-len(lines)/8 # if there's an even amount of frames i can use len(lines)-len(lines)/8
            fname = pathlib.Path(file).stem
            with open(f'{wrkdir}/{idx}/{fname}_{idx}.xvg', 'w') as f:
                f.writelines(lines[:idx])
            

    WHAM.WHAM(f'{wrkdir}/7500')
    WHAM.WHAM(f'{wrkdir}/5000')
    WHAM.WHAM(f'{wrkdir}/2500')

cooperativity_check(wrkdir)