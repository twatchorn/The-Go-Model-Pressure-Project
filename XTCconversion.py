import subprocess as sp
import os
import glob
import pathlib

#
# This script is for gmx trjconv for gromacs versions < 4.6 not using the gmx wrapper to take standard .xtc and .tpr files and generate a 'whole' molecule, remove pbc, and skip 100 frames
# Whole is a standard process that makes any molecules that "break" in the simulation one piece again
# nopbc is a process that recenters the molecule so the periodicity doesn't effect analysis
# skip is done primarily to condense the data enough to make it recognizable rather than trying to fit every frame into the analysis we can take a spread out sample of the data
# and use that to analyze native contacts and thermodynamics of the system
#


def xtcmods(wrkdir):
    
    def whole(xtc, tpr):
        file_name = os.path.basename(xtc).replace('.xtc', '_whole.xtc')
        output_path = f'{wrkdir}/{file_name}'
        
        process_whole = sp.Popen([
            "gmx", 'trjconv', 
            '-f', xtc, 
            '-s', tpr, 
            '-o', output_path, 
            '-pbc', 'whole'
        ], stdout=sp.PIPE, stdin=sp.PIPE, stderr=sp.PIPE, text=True)
        
        # Send group selection (assuming group 3, adjust as needed)
        stdout, stderr = process_whole.communicate(input="3\n")
        
        if process_whole.returncode != 0:
            with open(f'{wrkdir}/GMPP log.txt') as f:
                f.write(f"Error in whole step: {stderr}\n")
            return None
            
        return output_path

    def nopbc(xtc, tpr):
        file_name = os.path.basename(xtc).replace('.xtc', '_nopbc.xtc')
        output_path = f'{wrkdir}/{file_name}'
        
        process_nopbc = sp.Popen([
            "gmx", 'trjconv', 
            '-f', xtc, 
            '-s', tpr, 
            '-o', output_path, 
            '-pbc', 'mol', 
            '-center'
        ], stdout=sp.PIPE, stdin=sp.PIPE, stderr=sp.PIPE, text=True)
        
        # Send group selection for centering and output
        stdout, stderr = process_nopbc.communicate(input="3\n3\n")
        
        if process_nopbc.returncode != 0:
            with open(f'{wrkdir}/GMPP log.txt') as f:
                f.write(f"Error in nopbc step: {stderr}\n")
            return None
            
        return output_path

    def skip(xtc, tpr):
        file_name = os.path.basename(xtc).replace('.xtc', '_skip.xtc')
        output_path = f'{wrkdir}/{file_name}'
        
        process_skip = sp.Popen([
            'gmx', 'trjconv',  # Added 'gmx' prefix for consistency
            '-f', xtc, 
            '-s', tpr,  # Removed extra .tpr extension
            '-o', output_path, 
            '-skip', '100'
        ], stdout=sp.PIPE, stdin=sp.PIPE, stderr=sp.PIPE, text=True)
        
        stdout, stderr = process_skip.communicate(input="3\n")
        
        if process_skip.returncode != 0:
            with open(f'{wrkdir}/GMPP log.txt') as f:
                f.write(f"Error in skip step: {stderr}\n")
            return None
            
        return output_path

    # Process files
    xtc_files = glob.glob(f'{wrkdir}/*.xtc')
    tpr_files = glob.glob(f'{wrkdir}/*.tpr')
    
    for xtc_file in xtc_files:
        for tpr_file in tpr_files:
            # Match files by stem (filename without extension)
            if pathlib.Path(tpr_file).stem == pathlib.Path(xtc_file).stem:
                with open(f'{wrkdir}/GMPP log.txt') as f:
                    f.write(f"Processing {xtc_file} with {tpr_file}\n")
                
                # Step 1: Make molecules whole
                xtc_whole_path = whole(xtc_file, tpr_file)
                if not xtc_whole_path:
                    continue
                    
                # Step 2: Remove PBC and center
                xtc_nopbc_path = nopbc(xtc_whole_path, tpr_file)
                if not xtc_nopbc_path:
                    continue
                    
                # Step 3: Skip frames
                xtc_skip_path = skip(xtc_nopbc_path, tpr_file)
                if xtc_skip_path:
                    with open(f'{wrkdir}/GMPP log.txt') as f:
                        f.write(f"Successfully processed: {xtc_skip_path}\n")

