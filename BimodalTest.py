import seaborn as sns
import numpy as np
import pathlib
import glob

def bimodal(wdir):
        xvgs = glob.glob(f'{wdir}*.xvg')
        for file in xvgs:
            data = np.loadtxt(file, comments=['@', '#'], usecols=(1,))
            output = sns.histplot(data=data, stat='probability')
            patches = output.patches
            bin_edges = [patch.get_x() for patch in patches]
            ro = [patch.get_height() for patch in patches]

            for i, v in enumerate(ro):
                maxf = 0
                if v > maxf:
                    maxf = v
                    maxfi = i
            
            for i, v in enumerate(reversed(ro)):
                maxr = 0
                if v > maxr:
                    maxr = v
                    maxri = i

            if maxfi == maxri:
                modality = False
            else:
                if abs(maxfi - maxri) > 0.3:
                    modality = False
                else:
                    modality = True
            if modality == True:
                foldingtemp = int(pathlib.Path(file).stem) # extract temperature from file
                return foldingtemp