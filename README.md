# The-Go-Model-Pressure-Project
Official GMPP 

The Go-Model pressure project is designed to intake SMOG-server gromacs files and use a series of simulations to find the folding transition and develop a folding landscape for the given system. 

The landscapes will be generated with a series of combinations of the RMSF, native contacts, and radius of gyration with the dimensionless free energy calculated with the WHAM algorithm. 

Command line: python GMPP.py /path/to/smog/files.

Note: use tprgenie.py not tprgen.py, i will remove it soon i just want to ensure proper version handling first.
Example Landscapes for 1UBQ:
<img width="640" height="641" alt="image" src="https://github.com/user-attachments/assets/20c92fd3-6ef5-40fa-8d35-4afae9c2c77c" /> 
<img width="678" height="652" alt="image" src="https://github.com/user-attachments/assets/cb2d989e-c608-4e27-a61a-97ddf3205638" />
<img width="656" height="631" alt="image" src="https://github.com/user-attachments/assets/e88aea41-2638-4e1e-b504-169e36a85193" />




