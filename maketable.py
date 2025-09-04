# Table generator for gromacs V 4.x.x
# Based off of the smog-server maketable4.pl script
#
#
#
#
#
#
def tablegen(r):
    with open('table.xvg', 'w') as f:
        Rtable = r
        DR = 0.002
        Ntable = int(Rtable / DR)
        f.write("0.0 0.0 0.0 1.0 1.0 1.0 1.0")

        for i in range(1, Ntable):
            R = i * DR

            if R > 0.01:
                R1 = -1 / R**10
                R2 = -10 / R**11
                R3 = 1 / R**12
                R4 = 12.0 / R**13
                f.write(f"{R} 0.0 0.0 {R1} {R2} {R3} {R4}")
           
            else:
                f.write(f"{R} 0.0 0.0 1.0 1.0 1.0 1.0")
             


    
