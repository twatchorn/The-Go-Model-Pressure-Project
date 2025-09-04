import os
import subprocess
import glob
def tprgen(tlow, thigh, dt, wdir):  # Adde parameter
    
    # The preferred mdp settings for these coarse graining simulations are generated via the smog server 
    aa_mdp = glob.glob(f'{wdir}/*aa.mdp')
    ca_mdp = glob.glob(f'{wdir}/*ca.mdp')
    
    # Check if files exist
    """ if not aa_mdp or not ca_mdp:
        print("Error: Could not find aa.mdp or ca.mdp files")
        return """
    
    """ aa_mdp = aa_mdp[0]  # Get first match
    ca_mdp = ca_mdp[0] """  # Get first match

    lower_temp, upper_temp = tlow, thigh
    
    gro = glob.glob(f'{wdir}/*.gro')
    top = glob.glob(f'{wdir}/*.top')
    
    if not gro or not top:
        print("Error: Could not find .gro or .top files")
        return
        
    """ gro = gro[0]  # Get first match
    top = top[0]  # Get first matc """
    
    sim_type = 'ca'
    table = glob.glob(f'{wdir}/table.xvg')
    
    """ if table:
        table = table[0]
    else:
        table = "" """

    if sim_type == 'aa':
        with open(aa_mdp, 'r') as f:
            original_lines = f.readlines()
            
        for i in range(lower_temp, upper_temp+1, dt):
            if os.path.exists(f'{wdir}/{i}.tpr'):
                continue
                
            # Create modified lines for this temperature
            new_lines = []
            ref_t_found = False
            gen_temp_found = False
            
            for line in original_lines:
                if line.startswith('ref_t'):
                    new_lines.append(f'ref_t = {i}\n')
                    ref_t_found = True
                elif line.startswith('gen_temp'):
                    new_lines.append(f'gen_temp = {i}\n')
                    gen_temp_found = True
                else:
                    new_lines.append(line)

            if not ref_t_found:
                new_lines.append(f'ref_t = {i}\n')
            if not gen_temp_found:
                new_lines.append(f'gen_temp = {i}\n')

            # Write temporary mdp file
            temp_mdp = f'{wdir}/temp_{i}_aa.mdp'
            with open(temp_mdp, 'w') as f:
                f.writelines(new_lines)
                
            # Generate tpr
            cmd = f"grompp -f {temp_mdp} -c {gro} -p {top} -o {wdir}/{i}.tpr -maxwarn 1"
            try:
                tpr_pro = subprocess.check_call(cmd, shell=True)
            except subprocess.CalledProcessError as e:
                with open(f'{wdir}/GMPP log.txt', 'a') as f:
                    f.write(f"Error running GROMPP: {e}")
                    f.close()
            
            # Clean up temp file
            os.remove(temp_mdp)

    elif sim_type == 'ca':
        ca_mdp = ca_mdp[0]
        with open(ca_mdp, 'r') as f:
            original_lines = f.readlines()

        for i in range(lower_temp, upper_temp+1, dt):
            if os.path.exists(f'{wdir}/{i}.tpr'):
                continue
                
            # Create modified lines for this temperature
            new_lines = []
            ref_t_found = False
            gen_temp_found = False
            
            for line in original_lines:
                if line.startswith('ref_t'):
                    new_lines.append(f'ref_t = {i}\n')
                    ref_t_found = True
                elif line.startswith('gen_temp'):
                    new_lines.append(f'gen_temp = {i}\n')
                    gen_temp_found = True
                else:
                    new_lines.append(line)

            if not ref_t_found:
                new_lines.append(f'ref_t = {i}\n')
            if not gen_temp_found:
                new_lines.append(f'gen_temp = {i}\n')

            # Write temporary mdp file
            temp_mdp = f'{wdir}/temp_{i}_ca.mdp'
            with open(temp_mdp, 'w') as f:
                f.writelines(new_lines)
                
            # Generate tpr - FIXED: using ca_mdp instead of aa_mdp
            cmd = f"grompp -f {temp_mdp} -c {gro} -p {top} -o {wdir}/{i}.tpr -maxwarn 1"
            tpr_pro = subprocess.run(cmd, shell=True)
            
            # Clean up temp file
            os.remove(temp_mdp)

    # Run simulations
    tprfiles = glob.glob(f'{wdir}/*.tpr')
    
    for tpr_file in tprfiles:
        base_name = os.path.splitext(os.path.basename(tpr_file))[0]
        xtc_file = f'{wdir}/{base_name}.xtc'
        
        if os.path.exists(xtc_file):
            continue
        else:
            if table:
                cmd = f'mdrun -v -deffnm {wdir}/{base_name} -table {table} -tablep {table}'
            else:
                cmd = f'mdrun -v -deffnm {wdir}/{base_name}'
            process = subprocess.run(cmd, shell=True)

       