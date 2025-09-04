import numpy as np
import mdtraj as md
import glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------------
# Settings area for contact analysis
# use whichever pair selection you prefer, however i've seen a greater accuracy and much cleaner heatmaps for the probability using the smog server files 
# if you wish to use a standard mdtraj pair selection simply uncomment the next code block and modify the pair selections for your specific purposes. 
#heavy = native.topology.select_atom_indices('heavy')
#heavy_pairs = np.array(
#        [(i,j) for (i,j) in combinations(heavy, 2)
#            if abs(native.topology.atom(i).residue.index - \
#                   native.topology.atom(j).residue.index) > 3])
# The Best, Hummer function is used to calculate the probablity contact map and a seperate function is used to define Q(t) based off the contact.pl
# script from the Smog-Server. It is a rather robust analysis script that so far fits every system it's been applied to because it doesn't rely on fixed
# radii. The script extracts the native distances and finds a range for each distance designated by cut_off and ensures the resiudes are no closer than 70% the native range
#
#
#
# End Settings
#----------------------------------------------------------------------------------------------------------------
def contact_analysis(cut_off, wrkdir):
    cont_file = glob.glob(f'{wrkdir}/*.contacts')
    os.mkdir(f'{wrkdir}/Contact Plots', exist_ok = True)
    def best_hummer_q(traj, native, cont_file):
        """Compute the fraction of native contacts according the definition from
        Best, Hummer and Eaton [1]
        
        Parameters
        ----------
        traj : md.Trajectory
            The trajectory to do the computation for
        native : md.Trajectory
            The 'native state'. This can be an entire trajecory, or just a single frame.
            Only the first conformation is used
            
        Returns
        -------
        q : np.array, shape=(len(traj))
            The fraction of native contacts in each frame of `traj`
            
        References
        ----------
        ..[1] Best, Hummer, and Eaton, "Native contacts determine protein folding
            mechanisms in atomistic simulations" PNAS (2013)

        """

    
        beta_const = 50 # 1/nm
        lambda_const = 1.2 
        native_cutoff = 0.4 # nm
        
        
        
        df = pd.read_csv(cont_file, sep='\s+', usecols=[1, 3])
        
        capairs = df.values

        pairs_dist = md.compute_distances(native, capairs)
    
        native_contacts = capairs[np.any(pairs_dist < native_cutoff)]
        
        nativ2d = native_contacts.reshape(-1,2)
        
        r =  md.compute_distances(t, nativ2d)

        r0 = md.compute_distances(native[0], nativ2d)
        
        q = 1.0/(1+np.exp(beta_const*(r-lambda_const*r0)))
        
        return q, capairs

    #
    # qplot is a pretty easy to use manipulate function that can be modified to fit your needs
    # the basic idea is that the script will extract the first frame contact distances then produce 2 arrays, one that compares the distances to the maximum cutoff
    # of 120-140% of the native distances and one that compares the distances to the minimum cutoff of 70-80% of the native distances. If the distances is
    # less than the max cutoff and greater than the min cutoff its counted as a contact.
    #

    def qplot(traj, cont_file, cut_off):
        #prepping the cont file
        pairslist = pd.read_csv(cont_file, sep='\s+', usecols=[1, 3])
        capairs = pairslist.values
        natcounts = len(capairs)
        # extracting the native distances   
        natrng = md.compute_distances(t[0], capairs)
        if cut_off > 1:
            natmodu = natrng*cut_off
        else:
            natmodu = cut_off
        natmodl = natrng*.8
        # framewise distances
        conts = md.compute_distances(t, capairs)
        # compare the larger distances, if using hard cut off for all atom simulations (cut_off = int), for calpha simulations (cut_off = float) q will be calculated as a range relative to the native distance  
        frmcontsu = np.greater(natmodu, conts)
        frmcontsl = np.greater(conts, natmodl)
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



def contactgraphs(wrkdir, cut_off):
    xtc = glob.glob(f'{wrkdir}/MDOutputFiles/*.xtc')
    ca = glob.glob(f'{wrkdir}/MDOutputFiles/caonly.pdb')
    cont_file = glob.glob(f'{wrkdir}/*.contacts')

    for file in xtc:
        t = md.load(file, top='f{wrkdir}/caonly.pdb')
        q, capairs = contact_analysis.best_hummer_q(t, t[0], cont_file)
        dmap = md.geometry.squareform(q, capairs)
        cmap = np.mean(dmap, axis = 0)
        qt = contact_analysis.qplot(t, cont_file, cut_off)
        plt.clf()

        sns.heatmap(cmap, cmap = 'rocket_r', square=True)
        plt.gca().invert_yaxis()
        plt.xlabel('Residue #')
        plt.ylabel('Residue #')
        plt.savefig(f"{file} Probability Map.jpg")
        plt.clf()
        ymin, ymax = 0, 1

        plt.plot(qt, linewidth = 0.4)
        plt.ylim(ymin, ymax)
        plt.xlabel('Time (fs)')
        plt.ylabel('Q(t)')
        plt.title(f'Traditional Q from residues within 150% of native range, {file} K')
        plt.axhline(y=np.mean(qt), color='r', linestyle='--', label='Average')
        plt.savefig(f'{wrkdir}/Contact Plots/{file} Q(t).jpg')