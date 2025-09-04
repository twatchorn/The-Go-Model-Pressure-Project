import os
import mdtraj
import numpy as np


def coarspdb(workdir):
    def extract_ca(input_file, output_file):
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                if line.startswith('ATOM') and line[13:15] == 'CA':
                    outfile.write(line)


    input_file = glob.glob(f'{workdir}/*.pdb')
    output_file = f'{workdir}/caonly.pdb'

    extract_ca(input_file, output_file)