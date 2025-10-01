import os
import subprocess
import glob
import pathlib
import math as m

def tprgen(tlow, thigh, dt, wdir):
    log = f'{wdir}/GMPP log.txt'
    with open(log, 'a') as f:
        f.write(f'Starting tprgen with Temperature lower bound={tlow}, Temperature upper bound={thigh}\n')

    # The preferred mdp settings for these coarse graining simulations are generated via the smog server 
    # https://smog-server.org/prepare_a_simulation.html
    # 
    # !!!!!IMPORTANT!!!!!!
    # for the smog server coarse graining you will need to copy the .mdp settings from the smog server 
    # please title them ending with aa.mdp and ca.mdp respectively
    # cont_file refers to native contact files produced by coarse graining with the Smog-Server 
    # https://smog-server.org/prepare_a_simulation.html, 
    # although the data can be worked using the standard mdtraj pair selections
    
    aa_mdp = 'aa.mdp'
    ca_mdp = f'{wdir}/smogca.mdp'
    
    with open(log, 'a') as f:
        f.write(f'Using {ca_mdp}\n')

    # Temperature range will dictate the start and stopping points for what .tpr files will be generated
    # dt chooses the temperature step in kelvin between each .tpr file
    lower_temp, upper_temp = tlow, thigh

    gro = glob.glob(f'{wdir}/*.gro')
    top = glob.glob(f'{wdir}/*.top')
    
    if not gro:
        raise FileNotFoundError(f"No .gro file found in {wdir}")
    if not top:
        raise FileNotFoundError(f"No .top file found in {wdir}")

    sim_type = 'ca'
    gro = gro[0]
    top = top[0]
    
    # Read gro file and calculate box dimensions
    with open(f'{wdir}/GMPP log.txt', 'a') as f:
        f.write('Defining box dimensions for .mdp file\n')
    with open(gro, 'r') as f:
        lines = f.readlines()
        numat = lines[1][9:12].strip()  # Strip whitespace
        mind = float('inf')  # Initialize to infinity

    # Find the last line (box dimensions)
    last_line = lines[-1].strip()
    if last_line:  # Make sure it's not empty
        parts = last_line.split()
        if len(parts) >= 3:
            xdim = float(parts[0])
            ydim = float(parts[1]) 
            zdim = float(parts[2])
            
            # Calculate minimum distances
            zm = m.sqrt(xdim**2 + ydim**2)
            ym = m.sqrt(zdim**2 + xdim**2)
            xm = m.sqrt(zdim**2 + ydim**2)
            diag = m.sqrt(zdim**2+ydim**2+xdim**2) 
            mind = min(zm, ym, xm, diag)  # Much simpler way to find minimum
            maxd = max(zm, ym, xm, diag)
    blen = mind / 2.5

    with open(log, 'a') as f:
        f.write(f'.gro file used: {gro}\n.top file used: {top}\n')

    table = f'{wdir}/table.xvg'

    if sim_type == 'aa':
        for i in range(lower_temp, upper_temp+1, dt):
            if os.path.exists(f'{wdir}/{i}.tpr'):
                continue

            # Read current MDP file
            with open(aa_mdp, 'r') as f:
                lines = f.readlines()

            ref_t_found = False
            gen_temp_found = False
            new_lines = []

            for line in lines:
                if line.startswith('ref_t'):
                    new_lines.append(f'ref_t = {i}\n')
                    ref_t_found = True
                elif line.startswith('gen_temp'):
                    new_lines.append(f'gen_temp = {i}\n')
                    gen_temp_found = True
                else:
                    new_lines.append(line)

            # Add missing parameters only if they weren't found in the file
            if not ref_t_found:
                new_lines.append(f'ref_t = {i}\n')
            if not gen_temp_found:
                new_lines.append(f'gen_temp = {i}\n')

            # Write the modified MDP file
            with open(aa_mdp, 'w') as f:
                f.writelines(new_lines)

            # Generate TPR file
            result = subprocess.run(f"grompp -f {aa_mdp} -c {gro} -p {top} -o {wdir}/{i}.tpr -maxwarn 1", 
                                  shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                with open(log, 'a') as f:
                    f.write(f"Error generating TPR for {i}K: {result.stderr}\n")

        # Run simulations for AA
        tprfiles = glob.glob(f'{wdir}/*.tpr')
        for tpr_file in tprfiles:
            base_name = os.path.splitext(os.path.basename(tpr_file))[0]
            xtc_file = f'{wdir}/{base_name}.xtc'
            if os.path.exists(xtc_file):
                continue
            else:
                process = subprocess.run(f'mdrun -v -deffnm {base_name} -table {table} -tablep {table}', 
                                       shell=True, cwd=wdir, capture_output=True, text=True)

    elif sim_type == 'ca': 
        for i in range(lower_temp, upper_temp+1, dt):
            if os.path.exists(f'{wdir}/{i}.tpr'):
                continue

            # Read current MDP file
            with open(ca_mdp, 'r') as f:
                lines = f.readlines()

            ref_t_found = False
            gen_temp_found = False
            table_ext_found = False
            new_lines = []

            for line in lines:
                if line.startswith('ref_t'):
                    new_lines.append(f'ref_t = {i}\n')
                    ref_t_found = True
                elif line.startswith('gen_temp'):
                    new_lines.append(f'gen_temp = {i}\n')
                    gen_temp_found = True
                elif line.startswith('table-extension'):
                    new_lines.append(f'table-extension = {.25*blen}\n')
                    table_ext_found = True
                elif line.startswith('rlist'):
                    new_lines.append(f'rlist = {blen*.75}\n')
                elif line.startswith('rcoulomb'):
                    new_lines.append(f'rcoulomb = {blen*.75}\n')
                elif line.startswith('rvdw'):
                    new_lines.append(f'rvdw = {blen*.75}\n')
                else:
                    new_lines.append(line)

            # Add missing parameters only if they weren't found in the file
            if not ref_t_found:
                new_lines.append(f'ref_t = {i}\n')
            if not gen_temp_found:
                new_lines.append(f'gen_temp = {i}\n')
            if not table_ext_found:
                new_lines.append(f'table-extension = {blen}\n')

            # Write the modified MDP file
            with open(ca_mdp, 'w') as f:
                f.writelines(new_lines)

            # Generate TPR file
            result = subprocess.run(f"grompp -f {ca_mdp} -c {gro} -p {top} -o {wdir}/{i}.tpr -maxwarn 2", 
                                  shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                with open(log, 'a') as f:
                    f.write(f"Error generating TPR for {i}K: {result.stderr}\n")
                continue

        # Run all simulations after generating all TPR files
        tprfiles = glob.glob(f'{wdir}/*.tpr')
        with open(log, 'a') as f:
            f.write(f"Found {len(tprfiles)} TPR files to run\n")

        for tpr_file in tprfiles:
            base_name = os.path.splitext(os.path.basename(tpr_file))[0]
            xtc_file = f'{wdir}/{base_name}.xtc'
            
            if os.path.exists(xtc_file):
                with open(log, 'a') as f:
                    f.write(f"Skipping {base_name} - output already exists\n")
                continue

            with open(log, 'a') as f:
                f.write(f"Starting simulation for {base_name}\n")

            # Run each simulation and wait for completion
            result = subprocess.run(f'mdrun -v -deffnm {base_name} -table {table} -tablep {table}', 
                                  shell=True, cwd=wdir, capture_output=True, text=True)

            if result.returncode == 0:
                with open(log, 'a') as f:
                    f.write(f"Successfully completed {base_name}\n")
            else:
                with open(log, 'a') as f:
                    f.write(f"Error in {base_name}: {result.stderr}\n")
