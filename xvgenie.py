# xvg generator
import pexpect
import subprocess

def xvgenie(wdir):
    
    edrs = glob.glob(f'{wdir}/*.edr')
    for file in edrs:
        file = os.path.splitext(file)[0]
        # Define the GROMACS command to run
        gromacs_cmd = f"gmx energy -f {file}.edr -o potential_energy.xvg"

        # Spawn the GROMACS process using pexpect
        child = pexpect.spawn(gromacs_cmd)

        child.expect('              :-) GROMACS')
        child.sendline('Potential')
        # Wait for the process to finish
        child.expect(pexpect.EOF)

        # Get the output from the process
        output = child.before

        # Save the output to a file
        with open(f"{file}.xvg", "w") as f:
            f.write(output)