# xvg generator
import pexpect
import subprocess as sp
import glob
import os

def xvgenie(wdir):
    
    edrs = glob.glob(f'{wdir}/*.edr')
    for file in edrs:
        file = os.path.splitext(file)[0]
        # Define the GROMACS command to run
        gromacs_cmd = sp.Popen(["g_energy","-f",f"{file}.edr","-o",f"{file}_potential.xvg"], stdout=sp.PIPE, stdin=sp.PIPE, stderr=sp.PIPE, text=True)
        stdout, stderr = gromacs_cmd.communicate(input='Potential\n')

        if gromacs_cmd.returncode != 0:
            with open(f'{wdir}/GMPP log.txt','a') as f:
                f.write(f'Error in {stderr}')
