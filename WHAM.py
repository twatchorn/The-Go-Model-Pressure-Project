import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import math as m
import glob
import seaborn as sns
import pathlib   

def WHAM(wrkdir, tlow, thigh):
    xvg = glob.glob(f'{wrkdir}/*potential.xvg')
    
    def wham_iteration(histograms, bin_centers, bias_potentials, temperatures, max_iter=1000, tol=1e-8):
        """
        Standard WHAM implementation following Kumar et al. 1992
        
        Parameters:
        - histograms: list of histogram counts for each window [n_windows x n_bins]
        - bin_centers: energy values for histogram bins [n_bins]
        - bias_potentials: bias potential values at bin_centers for each window [n_windows x n_bins]
        - temperatures: temperature for each window [n_windows]
        - max_iter: maximum iterations
        - tol: convergence tolerance
        
        Returns:
        - f_k: converged free energies for each window
        - rho_unbiased: unbiased probability distribution
        """
        
        n_windows = len(histograms)
        n_bins = len(bin_centers)
        
        # Initialize free energies (first window = 0 by convention)
        f_k = np.zeros(n_windows)
        
        # Calculate beta values (1/kT) for each window
        kB = constants.Boltzmann * constants.Avogadro / 1000.0  # kJ/mol/K
        beta_k = 1.0 / (kB * np.array(temperatures))
        
        # Total counts in each window
        N_k = np.array([np.sum(hist) for hist in histograms])
       
        with open(f'{wrkdir}/GMPP log.txt', 'a') as f:
            f.write(f"Windows: {n_windows}, Bins: {n_bins}")
            f.write(f"Total counts per window: {N_k}")
        
        for iteration in range(max_iter):
            f_k_old = f_k.copy()
            
            # Calculate unbiased probability at each bin
            rho_unbiased = np.zeros(n_bins)
            
            for i in range(n_bins):
                numerator = 0.0
                denominator = 0.0
                
                # Sum over all windows for numerator
                for k in range(n_windows):
                    numerator += histograms[k][i]
                
                # Sum over all windows for denominator
                for k in range(n_windows):
                    if N_k[k] > 0:  # avoid division by zero
                        # For temperature replica exchange
                        bias_term = bin_centers[i] * (beta_k[k] - beta_k[0])
                        denominator += N_k[k] * np.exp(f_k[k] - bias_term)
                
                if denominator > 0:
                    rho_unbiased[i] = numerator / denominator
                else:
                    rho_unbiased[i] = 0.0
            
            # Update free energies using self-consistency equation
            for k in range(n_windows):
                if N_k[k] > 0:
                    sum_term = 0.0
                    for i in range(n_bins):
                        if rho_unbiased[i] > 0:
                            # For temperature replica exchange, the "bias" is the temperature difference
                            # U_bias = E * (beta_k - beta_0) where beta_0 is reference temperature
                            bias_term = bin_centers[i] * (beta_k[k] - beta_k[0])
                            sum_term += rho_unbiased[i] * np.exp(-bias_term)
                    
                    if sum_term > 0:
                        f_k[k] = -np.log(sum_term)
            
            # Set first window free energy to 0 (reference)
            f_k = f_k - f_k[0]
            
            # Check convergence
            if np.allclose(f_k, f_k_old, atol=tol) & np.any(f_k)>0:
                with open(f'{wrkdir}/GMPP log.txt', 'a') as f:

                    f.write(f'WHAM converged after {iteration + 1} iterations')
                    f.write(f'Final free energies: {f_k}')
                break
                
            if iteration == max_iter - 1:
                with open(f'{wrkdir}/GMPP log.txt', 'a') as f:
                    f.write('WHAM did not converge within maximum iterations')
        
        return f_k, rho_unbiased

    def calculate_partition_function(rho_unbiased, bin_centers, temperature, f_k_ref=0.0):
        """
        Calculate partition function from unbiased probability distribution
        
        Parameters:
        - rho_unbiased: unbiased probability distribution
        - bin_centers: energy values 
        - temperature: temperature for calculation
        - f_k_ref: free energy of reference state (usually 0)
        """
        kB = constants.Boltzmann * constants.Avogadro / 1000.0  # kJ/mol/K
        beta = 1.0 / (kB * temperature)
        
        # The density of states g(E) is related to the unbiased probability by:
        # rho(E) = g(E) * exp(-beta*E - f_ref) / Z
        # Therefore: g(E) = rho(E) * Z * exp(beta*E + f_ref)
        # But we can work directly with the unbiased probabilities
        
        bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0
        
        # Partition function: Z = sum over E of g(E) * exp(-beta*E)
        # Since rho(E) = g(E)*exp(-beta*E)/Z, we have:
        # Z = sum over E of rho(E) * Z * exp(beta*E) * exp(-beta*E) = Z * sum(rho(E))
        # So Z cancels out and Z = sum(rho(E)) * normalization_factor
        
        # More directly: the degeneracy is proportional to rho(E) * exp(beta*E)
        degeneracy = rho_unbiased * np.exp(beta * bin_centers + f_k_ref)
        
        # Partition function
        partition_function = np.sum(degeneracy * np.exp(-beta * bin_centers)) * bin_width
        
        return partition_function, degeneracy

    def calculate_heat_capacity(bin_centers, degeneracy, temp_range):
        """
        Calculate heat capacity as a function of temperature using the density of states
        
        Cv = (1/kT^2) * (<E^2> - <E>^2)
        
        Parameters:
        - bin_centers: energy values
        - degeneracy: density of states g(E)
        - temp_range: array of temperatures to calculate Cv for
        
        Returns:
        - temperatures, heat_capacities, energy_expectations, energy_variances
        """
        kB = constants.Boltzmann * constants.Avogadro / 1000.0  # kJ/mol/K
        
        heat_capacities = []
        energy_expectations = []
        energy_variances = []
        
        bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0
        
        for T in temp_range:
            beta = 1.0 / (kB * T)
            
            # Calculate partition function at this temperature
            boltzmann_factors = np.exp(-beta * bin_centers)
            Z = np.sum(degeneracy * boltzmann_factors) * bin_width
            
            if Z > 0:
                # Calculate probabilities at this temperature
                prob_T = (degeneracy * boltzmann_factors) / Z
                
                # Energy expectation: <E> = sum(E * P(E))
                E_avg = np.sum(bin_centers * prob_T) * bin_width
                
                # Energy squared expectation: <E^2> = sum(E^2 * P(E))
                E2_avg = np.sum(bin_centers**2 * prob_T) * bin_width
                
                # Energy variance: <E^2> - <E>^2
                E_var = E2_avg - E_avg**2
                
                # Heat capacity: Cv = (1/kT^2) * Var(E)
                Cv = E_var / (kB * T**2)
                
                heat_capacities.append(Cv)
                energy_expectations.append(E_avg)
                energy_variances.append(E_var)
            else:
                heat_capacities.append(0.0)
                energy_expectations.append(0.0)
                energy_variances.append(0.0)
        
        return np.array(temp_range), np.array(heat_capacities), np.array(energy_expectations), np.array(energy_variances)

    def process_wham_data(xvg_files, n_bins=50):
        """
        Main function to process WHAM data from XVG files with heat capacity analysis
        """
        
        temperatures = np.zeros(len(xvg_files))
        histograms = []
        all_energies = []
        
        # Read data from all files
        for file in xvg_files:
            # Extract temperature from filename
            temp = pathlib.Path(file).stem.split('_')[0]
        
            np.append(temperatures, temp, axis=None)   
            
            
            # Load potential energy data
            potentials = np.loadtxt(file, comments=['@', '#'], usecols=[1])
            potentials = potentials[1:]  # skip first point if needed
            all_energies.extend(potentials)
            
            with open(f'{wrkdir}/GMPP log.txt', 'a') as f:
                f.write(f'Loaded {len(potentials)} points from T={temp}K')
        
        # Create temperature range for heat capacity from the simulation temperatures
        T_min, T_max = tlow, thigh
        # Create a smooth range with more points for better resolution
        temp_range = np.linspace(T_min * 0.9, T_max * 1.1, 200)
        
        # Create common bin edges for all histograms
        e_min, e_max = np.min(all_energies), np.max(all_energies)
        bin_edges = np.linspace(e_min, e_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        with open(f'{wrkdir}/GMPP log.txt', 'a') as f:    
            f.write(f'Energy range: {e_min:.2f} to {e_max:.2f} kJ/mol')
            f.write(f'Bin width: {(e_max - e_min)/n_bins:.3f} kJ/mol')
        
        # Create histograms for each window
        for i, file in enumerate(xvg_files):
            potentials = np.loadtxt(file, comments=['@', '#'], usecols=[1])
            potentials = potentials[1:]
            
            hist, _ = np.histogram(potentials, bins=bin_edges)
            histograms.append(hist)
        
        # For this example, assuming no additional bias potentials (just temperature)
        # If you have umbrella sampling or other biases, you'd calculate them here
        bias_potentials = np.zeros((len(temperatures), len(bin_centers)))
        
        # Run WHAM
        f_k, rho_unbiased = wham_iteration(histograms, bin_centers, bias_potentials, temperatures)
        
        # Calculate partition function at reference temperature
        ref_temp = temperatures[0]  # or choose your reference
        Z, degeneracy = calculate_partition_function(rho_unbiased, bin_centers, ref_temp, f_k[0])
        
        with open(f'{wrkdir}/GMPP log.txt', 'a') as f:
            f.write(f'Free energies: {f_k}')
            f.write(f'Partition function at {ref_temp}K: {Z:.6e}')
        
        # Calculate heat capacity over temperature range
        temps, heat_caps, energy_avgs, energy_vars = calculate_heat_capacity(bin_centers, degeneracy, temp_range)
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        
        
        # Bottom row: Thermodynamic properties
        plt.subplot(2, 3, 1)
        plt.plot(temps, energy_avgs, 'b-', linewidth=2, label='<E>')
        plt.scatter(temperatures, [np.mean(np.loadtxt(f, comments=['@', '#'], usecols=[1])[1:]) 
                                for f in xvg_files], c='red', s=50, zorder=5, label='Simulation <E>')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Energy Expectation (kJ/mol)')
        plt.title('Energy vs Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        plt.plot(temps, heat_caps, 'g-', linewidth=2)
        plt.xlabel('Temperature (K)')
        plt.ylabel('Heat Capacity (kJ/mol/K)')
        plt.title('Heat Capacity vs Temperature')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 6)
        plt.plot(temps, energy_vars, 'orange', linewidth=2)
        plt.xlabel('Temperature (K)')
        plt.ylabel('Energy Variance (kJ²/mol²)')
        plt.title('Energy Fluctuations vs Temperature')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plt.savefig(f'{wrkdir}/MDOutputFiles/WHAM.png')

    
        max_cv_idx = np.argmax(heat_caps)
        with open(f'{wrkdir}/GMPP log.txt', 'a') as f:
            f.write(f'\nThermodynamic Analysis:')
            f.write(f'Maximum heat capacity: {heat_caps[max_cv_idx]:.4f} kJ/mol/K at {temps[max_cv_idx]:.1f} K')
            f.write(f'Energy range: {energy_avgs.min():.2f} to {energy_avgs.max():.2f} kJ/mol')
            f.close()
        
        return f_k, rho_unbiased, bin_centers, Z, degeneracy, temps, heat_caps, energy_avgs


    f_k, rho_unbiased, bin_centers, Z, degeneracy, temps, heat_caps, energy_avgs = process_wham_data(xvg)
