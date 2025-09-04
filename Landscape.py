import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
import pathlib
import mdtraj as md
import seaborn as sns
import glob
import pandas as pd
import os



def landscape(wrkdir, ft, cut_off, cont_file):
    def qplot(traj, cont_file, cut_off):
    #prepping the cont file
        pairslist = pd.read_csv(cont_file, sep='\s+', usecols=[1, 3])
        capairs = pairslist.values
        natcounts = len(capairs)
        # extracting the native distances   
        natrng = md.compute_distances(traj[0], capairs)
        if cut_off > 1:
            natmodu = natrng*cut_off
        else:
            natmodu = cut_off
        natmodl = natrng*.8
        # framewise distances
        conts = md.compute_distances(traj, capairs)
        # compare the larger distances, if using hard cut off for all atom simulations (cut_off = int), for calpha simulations (cut_off = float) q will be calculated as a range relative to the native distance  
        frmcontsu = np.greater(natmodu, conts)
        frmcontsl= np.greater(conts, natmodl)
        #comparing the 2 arrays
        result = np.equal(frmcontsu, frmcontsl)
        frmcounts = []
        counts = 0
        # simple loop to go from a boolean array to a Q(t) array
        for row in result:
            for element in row:
                if element == True:
                    counts += 1
            frmcounts.append(counts)
            counts = 0

        qt = []
        qf = 0
        for i in frmcounts:
            qf = i/natcounts
            qt.append(qf)
            qf = 0
        return qt
    

    def extract_multiple_reference_structures(wrkdir):
        """
        Extract multiple candidate reference structures using different methods
        """

        print(f"{'='*60}")
        print("EXTRACTING MULTIPLE REFERENCE STRUCTURES")
        print(f"{'='*60}")

        references = {}
        pdb_file = f'{wrkdir}/caonly.pdb'

        # Find all temperature simulations
        xtc_files = glob.glob(f'{wrkdir}/*skip.xtc')
        temp_files = {}

        for xtc_file in xtc_files:
            try:
                temp = int(pathlib.Path(xtc_file).stem.split('_')[0])
                temp_files[temp] = xtc_file
            except ValueError:
                continue
            
        if not temp_files:
            print("No temperature-named XTC files found!")
            return {}

        temperatures = sorted(temp_files.keys())
        print(f"Found temperatures: {temperatures}")

        # Method 1: Coldest simulation - highest Q frame
        print(f"\n{'-'*40}")
        print("METHOD 1: Highest Q frame from coldest simulation")
        print(f"{'-'*40}")

        try:
            coldest_temp = min(temperatures)
            cold_traj = md.load(temp_files[coldest_temp], top=pdb_file)

            # Calculate Q for cold simulation using middle frame as initial reference
            contfile = glob.glob(f'{wrkdir}/*.contacts')[0]
            contacts_data = pd.read_csv(contfile, sep='\\s+', skiprows=1)
            pairs = contacts_data.iloc[:, [1, 3]].values - 1
            pairs = pairs.astype(int)

            mid_frame = cold_traj.n_frames // 2
            native_distances = md.compute_distances(cold_traj[mid_frame:mid_frame+1], pairs)[0]

            # Calculate Q for all frames
            frame_distances = md.compute_distances(cold_traj, pairs)
            upper_cutoff = native_distances * 1.2
            lower_cutoff = native_distances * 0.8
            within_upper = frame_distances <= upper_cutoff
            above_lower = frame_distances >= lower_cutoff
            native_contacts = within_upper & above_lower
            q_values = np.mean(native_contacts, axis=1)

            max_q_frame = np.argmax(q_values)
            references[f'cold_max_Q'] = {
                'traj': cold_traj,
                'frame': max_q_frame,
                'temp': coldest_temp,
                'Q': q_values[max_q_frame],
                'method': f'Highest Q frame ({max_q_frame}) from {coldest_temp}K'
            }

            print(f"✓ Coldest ({coldest_temp}K) max Q frame: {max_q_frame}, Q = {q_values[max_q_frame]:.3f}")

        except Exception as e:
            print(f"✗ Error with coldest simulation method: {e}")

        # Method 2: Minimum radius of gyration from coldest simulation
        print(f"\n{'-'*40}")
        print("METHOD 2: Minimum Rg frame from coldest simulation")
        print(f"{'-'*40}")

        try:
            rg_values = md.compute_rg(cold_traj)
            min_rg_frame = np.argmin(rg_values)

            references[f'cold_min_Rg'] = {
                'traj': cold_traj,
                'frame': min_rg_frame,
                'temp': coldest_temp,
                'Rg': rg_values[min_rg_frame],
                'method': f'Minimum Rg frame ({min_rg_frame}) from {coldest_temp}K'
            }

            print(f"✓ Coldest ({coldest_temp}K) min Rg frame: {min_rg_frame}, Rg = {rg_values[min_rg_frame]*10:.2f} Å")

        except Exception as e:
            print(f"✗ Error with min Rg method: {e}")

        # Method 3: Starting structure (.gro file)
        print(f"\n{'-'*40}")
        print("METHOD 3: Starting structure (.gro file)")
        print(f"{'-'*40}")

        try:
            gro_files = glob.glob(f'{wrkdir}/*.gro')
            if gro_files:
                gro_traj = md.load(gro_files[0])

                references['gro_start'] = {
                    'traj': gro_traj,
                    'frame': 0,
                    'temp': 'start',
                    'method': f'Starting structure from {os.path.basename(gro_files[0])}'
                }

                print(f"✓ Starting structure: {os.path.basename(gro_files[0])}")
            else:
                print("✗ No .gro file found")

        except Exception as e:
            print(f"✗ Error loading .gro file: {e}")

        # Method 4: Minimum energy frame (if energy files available)
        print(f"\n{'-'*40}")
        print("METHOD 4: Minimum energy frame")
        print(f"{'-'*40}")

        for temp in temperatures[:3]:  # Check first few temperatures
            try:
                energy_file = f'{wrkdir}/{temp}_potential.xvg'
                if os.path.exists(energy_file):
                    energy_data = np.loadtxt(energy_file, comments=['@', '#'])
                    energies = energy_data[:, 1]

                    traj = md.load(temp_files[temp], top=pdb_file)
                    min_frames = len(energies) if len(energies) < traj.n_frames else traj.n_frames

                    min_energy_frame = np.argmin(energies[:min_frames])

                    references[f'T{temp}_min_E'] = {
                        'traj': traj,
                        'frame': min_energy_frame,
                        'temp': temp,
                        'energy': energies[min_energy_frame],
                        'method': f'Min energy frame ({min_energy_frame}) from {temp}K'
                    }

                    print(f"✓ T={temp}K min energy frame: {min_energy_frame}, E = {energies[min_energy_frame]:.2f}")
                    break

            except Exception as e:
                print(f"✗ Error with energy method for T={temp}K: {e}")

        # Method 5: Most compact structure across all temperatures
        print(f"\n{'-'*40}")
        print("METHOD 5: Most compact structure across all temperatures")
        print(f"{'-'*40}")

        try:
            global_min_rg = float('inf')
            best_compact = None

            for temp in temperatures[:3]:  # Check first few temperatures
                traj = md.load(temp_files[temp], top=pdb_file)
                rg_vals = md.compute_rg(traj)
                min_rg_idx = np.argmin(rg_vals)
                min_rg_val = rg_vals[min_rg_idx]

                if min_rg_val < global_min_rg:
                    global_min_rg = min_rg_val
                    best_compact = {
                        'traj': traj,
                        'frame': min_rg_idx,
                        'temp': temp,
                        'Rg': min_rg_val,
                        'method': f'Global min Rg frame ({min_rg_idx}) from {temp}K'
                    }

            if best_compact:
                references['global_min_Rg'] = best_compact
                print(f"✓ Global most compact: T={best_compact['temp']}K, frame {best_compact['frame']}, "
                      f"Rg = {best_compact['Rg']*10:.2f} Å")

        except Exception as e:
            print(f"✗ Error with global compact method: {e}")

        # Method 6: Maximum contact formation frame
        print(f"\n{'-'*40}")
        print("METHOD 6: Maximum native contact formation")
        print(f"{'-'*40}")

        try:
            # Use first frame of coldest as initial reference, then find frame with most contacts
            first_distances = md.compute_distances(cold_traj[0:1], pairs)[0]
            frame_distances = md.compute_distances(cold_traj, pairs)

            # Count contacts within reasonable distance
            contact_cutoff = 0.8  # nm, typical for CA contacts
            contacts_per_frame = np.sum(frame_distances < contact_cutoff, axis=1)
            max_contact_frame = np.argmax(contacts_per_frame)

            references['max_contacts'] = {
                'traj': cold_traj,
                'frame': max_contact_frame,
                'temp': coldest_temp,
                'contacts': contacts_per_frame[max_contact_frame],
                'method': f'Max contacts frame ({max_contact_frame}) from {coldest_temp}K'
            }

            print(f"✓ Max contacts: frame {max_contact_frame}, {contacts_per_frame[max_contact_frame]} contacts")

        except Exception as e:
            print(f"✗ Error with max contacts method: {e}")

        # Save all reference structures as PDB files
        print(f"\n{'-'*40}")
        print("SAVING REFERENCE STRUCTURES")
        print(f"{'-'*40}")

        saved_refs = {}
        for ref_name, ref_data in references.items():
            try:
                pdb_filename = f'{wrkdir}/ref_{ref_name}.pdb'
                ref_data['traj'][ref_data['frame']].save_pdb(pdb_filename)
                saved_refs[ref_name] = {**ref_data, 'pdb_file': pdb_filename}
                print(f"✓ Saved {ref_name}: {pdb_filename}")
            except Exception as e:
                print(f"✗ Error saving {ref_name}: {e}")

        print(f"\nTotal reference structures extracted: {len(saved_refs)}")
        return saved_refs

    def parse_xvg_file(xvg_file):
        """
        Parse GROMACS .xvg output file
        """
        data = []
        with open(xvg_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith('#') or line.startswith('@') or not line:
                    continue
                # Parse data lines
                values = line.split()
                data.append([float(v) for v in values])

        return np.array(data)
    def calculate_all_metrics_combinations(wrkdir, folding_temp, cutoff, references):
        """
        Calculate Q(t), RMSD, and Rg using all reference combinations
        Modified to use GROMACS for RMSD calculation
        """

        print(f"{'='*60}")
        print(f"CALCULATING ALL METRICS FOR T={folding_temp}K")
        print(f"{'='*60}")

        # Load target trajectory
        target_traj_file = f'{wrkdir}/{folding_temp}_whole_nopbc.xtc'
        pdb_file = f'{wrkdir}/caonly.pdb'
        tpr_file = f'{wrkdir}/{folding_temp}.tpr'

        # Check if TPR file exists
        if not os.path.exists(tpr_file):
            # Try to find any TPR file in the directory
            tpr_files = glob.glob(f'{wrkdir}/*.tpr')
            if tpr_files:
                tpr_file = tpr_files[0]
                print(f"Using TPR file: {tpr_file}")
            else:
                print("✗ No TPR file found for GROMACS RMSD calculation")
                return None

        try:
            target_traj = md.load(target_traj_file, top=pdb_file)
            print(f"✓ Loaded target trajectory: {target_traj.n_frames} frames")
        except Exception as e:
            print(f"✗ Error loading target trajectory: {e}")
            return None

        # Calculate Rg for target 
        rg_target = md.compute_rg(target_traj)  
        print(f"✓ Calculated Rg: {np.mean(rg_target):.2f}±{np.std(rg_target):.2f} Å")

        results = {}

        # Load contacts for Q calculation
        try:
            contfile = glob.glob(f'{wrkdir}/*.contacts')[0]
            contacts_data = pd.read_csv(contfile, sep='\\s+', skiprows=1)
            pairs = contacts_data.iloc[:, [1, 3]].values - 1
            pairs = pairs.astype(int)
        except (IndexError, FileNotFoundError):
            print("✗ No contacts file found")
            return None

        for ref_name, ref_data in references.items():
            print(f"\n{'-'*30}")
            print(f"REFERENCE: {ref_name}")
            print(f"Method: {ref_data['method']}")
            print(f"{'-'*30}")

            try:
                # Calculate Q(t) using this reference (keeping MDTraj for this)
                ref_traj = ref_data['traj']
                ref_frame = ref_data['frame']

                q = qplot(target_traj, contfile, cutoff)
                q_values = np.array(q)
                print(f'Successfully calculated Q(T) for {target_traj}')

                rmsd_values = md.rmsd(target_traj, ref_traj, frame=ref_frame)
                print(f'Successfully calculated RMSD for {target_traj}')

                results[ref_name] = {
                    'Q': q_values,
                    'RMSD': rmsd_values,
                    'Rg': rg_target,
                    'reference_info': ref_data
                }



            except Exception as e:
                print(f"✗ Error calculating metrics for {ref_name}: {e}")

        return results
    def create_landscape(wrkdir, folding_temp, results):
        """
        Create proper 2D free energy landscapes F(Q,RMSD), F(Q,Rg), F(RMSD,Rg)
        Like the ones shown in the ubiquitin/lysozyme examples
        """

        print(f"{'='*60}")
        print("CREATING 2D FREE ENERGY LANDSCAPES F(Q,RMSD)")
        print(f"{'='*60}")

        # Define the landscape combinations  
        # cmap ex = ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])
        """ cmap_1 = ListedColormap(['yellow', 'lime', 'teal', 'mediumslateblue', 'indigo'])
        cmap_2 = ListedColormap(['yellow', 'rosybrown', 'tomato', 'firebrick', 'maroon'])
        cmap_3 = ListedColormap(['yellow', 'orange','tan','silver','']) """

        landscape_combinations = [
            ('Q', 'RMSD', 'Q(t)', 'RMSD (Å)', 'viridis'),
            ('Q', 'Rg', 'Q(t)', 'Rg (Å)', 'magma'), 
            ('RMSD', 'Rg', 'RMSD (Å)', 'Rg (Å)', 'cividis')
        ] # plasma to inferno

        n_refs = len(results)
        n_combos = len(landscape_combinations)

        # Create figure
        fig, axes = plt.subplots(n_combos, n_refs, figsize=(6*n_refs, 6*n_combos))
        if n_refs == 1:
            axes = axes.reshape(-1, 1)
        if n_combos == 1:
            axes = axes.reshape(1, -1)

        for combo_idx, (x_metric, y_metric, x_label, y_label, colormap) in enumerate(landscape_combinations):
            for ref_idx, (ref_name, ref_results) in enumerate(results.items()):

                ax = axes[combo_idx, ref_idx]

                try:
                    # Get the data
                    x_data = ref_results[x_metric]
                    y_data = ref_results[y_metric]

                    min_len = min(len(x_data), len(y_data))
                    x_data = x_data[:min_len]
                    y_data = y_data[:min_len]

                    # Debug: Check Q range
                    if x_metric == 'Q':
                        print(f"Q range for {ref_name}: {np.min(x_data):.3f} - {np.max(x_data):.3f}")
                    if y_metric == 'Q':
                        print(f"Q range for {ref_name}: {np.min(y_data):.3f} - {np.max(y_data):.3f}")

                    # Create 2D histogram P(Q,RMSD)
                    n_bins = 50
                    hist_2d, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=n_bins)

                    # Remove empty bins and normalize to get probability
                    total_counts = np.sum(hist_2d)
                    prob_2d = hist_2d / total_counts

                    # Convert to free energy F = -kT * ln(P)
                    # Set minimum probability to avoid log(0)
                    min_prob = 1e-8  # Very small but not zero
                    prob_2d = np.maximum(prob_2d, min_prob)

                    # Calculate free energy (in units of kT)
                    free_energy_2d = -np.log(prob_2d)

                    # Set the minimum free energy to 0 (reference state)
                    min_fe = np.min(free_energy_2d)
                    free_energy_2d = free_energy_2d - min_fe

                    # Set a reasonable maximum for visualization
                    max_fe = np.percentile(free_energy_2d, 100)  # Use 95th percentile as max
                    free_energy_2d = np.minimum(free_energy_2d, max_fe)

                    # Plot the free energy landscape
                    im = ax.imshow(free_energy_2d.T, origin='lower', aspect='auto',
                                  extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                                  cmap=colormap, vmin=0, vmax=max_fe)

                    # Add contour lines
                    X, Y = np.meshgrid((x_edges[:-1] + x_edges[1:])/2, (y_edges[:-1] + y_edges[1:])/2)
                    contour_levels = np.linspace(0, max_fe, 8)
                    ax.contour(X, Y, free_energy_2d.T, levels=contour_levels, 
                              colors='white', alpha=0.7, linewidths=1.0)

                    # Labels and title
                    ax.set_xlabel(x_label, fontsize=12)
                    ax.set_ylabel(y_label, fontsize=12)




                    title = f'{ref_name}\nF({x_metric},{y_metric})\n'
                    ax.set_title(title, fontsize=11)

                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, label='Free Energy F (kT)', shrink=0.8)
                    cbar.ax.tick_params(labelsize=10)

                    # Print landscape statistics
                    print(f"\n{ref_name} - {x_metric} vs {y_metric}:")
                    print(f"  Free energy range: 0 - {max_fe:.1f} kT")
                    print(f"  Number of sampled bins: {np.sum(hist_2d > 0)}/{n_bins*n_bins}")

                    # Find and report minima
                    for x, y in zip(x_data, y_data):
                        folded_mask = (x > 0.7) if x == 'Q' else (y_data > 0.7)
                        unfolded_mask = (x < 0.3) if x == 'Q' else (y_data < 0.3)

                    if np.any(folded_mask) and np.any(unfolded_mask):
                        folded_rmsd = y_data[folded_mask] if y_metric == 'RMSD' else x_data[folded_mask]
                        unfolded_rmsd = y_data[unfolded_mask] if y_metric == 'RMSD' else x_data[unfolded_mask]

                        print(f"  Folded state (Q>0.7): RMSD = {np.mean(folded_rmsd):.2f}±{np.std(folded_rmsd):.2f} Å")
                        print(f"  Unfolded state (Q<0.3): RMSD = {np.mean(unfolded_rmsd):.2f}±{np.std(unfolded_rmsd):.2f} Å")

                    # Save data
                    np.savez(f"{wrkdir}/MDOutputFiles/{folding_temp}K_{ref_name}_{x_metric}_{y_metric}_landscape.npz",
                             x_data=x_data, y_data=y_data, 
                             free_energy_2d=free_energy_2d,
                             x_edges=x_edges, y_edges=y_edges, 
                             hist_2d=hist_2d, prob_2d=prob_2d)

                except Exception as e:
                    print(f"✗ Error for {ref_name} {x_metric} vs {y_metric}: {e}")
                    ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=8, color='red')
                    ax.set_title(f'{ref_name}: {x_metric} vs {y_metric}\nERROR', color='red')

        plt.tight_layout()
        plt.savefig(f"{wrkdir}/MDOutputFiles/{folding_temp}K_2D_Free_Energy_Landscapes.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Create individual high-quality plots for each reference 
        for ref_name, ref_results in results.items():

            # Focus on Q vs RMSD (the classic folding landscape)
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            try:
                q_data = ref_results['Q']
                rmsd_data = ref_results['RMSD']

                min_len = min(len(q_data), len(rmsd_data))
                q_data = q_data[:min_len]
                rmsd_data = rmsd_data[:min_len]

                # High resolution 2D histogram
                n_bins = 50
                hist_2d, q_edges, rmsd_edges = np.histogram2d(q_data, rmsd_data, bins=n_bins)

                # Convert to free energy
                total_counts = np.sum(hist_2d)
                prob_2d = hist_2d / total_counts
                prob_2d = np.maximum(prob_2d, 1e-8)

                free_energy_2d = -np.log(prob_2d)
                min_fe = np.min(free_energy_2d)
                free_energy_2d = free_energy_2d - min_fe

                # Plot with nice colors (like the slides)
                max_fe = np.percentile(free_energy_2d, 100)
                free_energy_2d = np.minimum(free_energy_2d, max_fe)

                im = ax.imshow(free_energy_2d.T, origin='lower', aspect='auto',
                              extent=[q_edges[0], q_edges[-1], rmsd_edges[0], rmsd_edges[-1]],
                              cmap='jet', vmin=0, vmax=max_fe)

                # Add contours
                Q_centers = (q_edges[:-1] + q_edges[1:])/2
                RMSD_centers = (rmsd_edges[:-1] + rmsd_edges[1:])/2
                X, Y = np.meshgrid(Q_centers, RMSD_centers)

                contour_levels = np.linspace(0, max_fe, 10)
                ax.contour(X, Y, free_energy_2d.T, levels=contour_levels, 
                          colors='black', alpha=0.8, linewidths=1.2)

                ax.set_xlabel('Native contacts (Q)', fontsize=14)
                ax.set_ylabel('RMSD (Å)', fontsize=14)
                ax.set_title(f'Free Energy Landscape F(Q,RMSD)\n{ref_name} at T = {folding_temp}K', fontsize=16)

                # Nice colorbar
                cbar = plt.colorbar(im, ax=ax, label='Free Energy (kT)')
                cbar.ax.tick_params(labelsize=12)

                plt.tight_layout()
                plt.savefig(f"{wrkdir}/MDOutputFiles/{folding_temp}K_{ref_name}_Q_RMSD_Landscape_HighRes.png", 
                            dpi=300, bbox_inches='tight')
                plt.close()

            except Exception as e:
                print(f"✗ Error creating high-res plot for {ref_name}: {e}")

        print(f"\n{'='*50}")
        print("2D FREE ENERGY LANDSCAPES COMPLETE!")
        print(f"{'='*50}")

        return results
    def comprehensive_landscape_analysis(wrkdir, folding_temp, cutoff):
        """
        Complete analysis with correct 2D free energy landscapes
        """

        print(f"{'='*70}")
        print(f"COMPREHENSIVE LANDSCAPE ANALYSIS")
        print(f"Target Temperature: {folding_temp}K")
        print(f"Cutoff: {cutoff}")
        print(f"{'='*70}")

        # Step 1: Extract multiple reference structures
        references = extract_multiple_reference_structures(wrkdir)

        if not references:
            print("No reference structures could be extracted!")
            return None

        # Step 2: Calculate all metrics for all references
        results = calculate_all_metrics_combinations(wrkdir, folding_temp, cutoff, references)

        if not results:
            print("No metrics could be calculated!")
            return None

        # Step 3: Create correct 2D free energy landscapes
        results = create_landscape(wrkdir, folding_temp, results)

        return results

    analysis_results = comprehensive_landscape_analysis(wrkdir, ft, cut_off)
