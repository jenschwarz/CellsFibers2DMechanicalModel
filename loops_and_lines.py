"""
Created:       October 2017
Authors:       Tyler A. Engstrom (initiated code), Mahesh C. Gandikota, J. M. Schwarz
Description:   Generates and deforms a 2D semiflexible polymer network with area-preserving inclusions 

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as col

class RandomSpringNetwork:
    
    
    def __init__(self, xSize=4, ySize=6, orientation=1):
        """
        Class constructor. xSize can be any positive integer, while ySize must 
        be even. orientation should have a value of 0 or 1.
        """
        if ySize%2 != 0: 
            print('ySize must be even')
        if (orientation==0 or orientation==1) == False: 
            print('invalid orientation')
        self.Nx = xSize
        self.Ny = ySize
        self.orient = orientation
        N = xSize*ySize
        self.N = N
        self.X = np.zeros((N,2))  # cartesian coords (x, y) of the spring nodes
        self.R = np.zeros((N,2))  # cyl coords (z, phi) of the spring nodes
        self.R_0 = np.zeros((N,2)) # to store initial config
        self.cylRad   = 0.0
        self.cylRad_0 = 0.0
        self.cylLen   = 0.0
        self.cylLen_0 = 0.0
        self.K = np.zeros((N,N))  # symmetric matrix of spring constants 
        self.K_PBC = np.zeros((N,N))  # symmetric matrix of spring constants
        self.K_hookean = 0            #hookean spring constant
        self.K_sfp = 0            #sfp angular spring constant
        self.K_cross = 0          #crosslink angular spring constant
        self.K_area = 0
        self.area0 = np.sqrt(3)/4
        self.num_of_incl = 0      #number of inclusions        
        self.position = []       #positions of inclusions
        self.orientation = []    #orientations of inclusions
        self.strain = 0.0
        self.X_edges=np.zeros((N,2))      
        
    def count(self):
        """
        to count number of springs in system
        """
        N = self.N
        
        k = 0
        for i in range(N):
            for j in range(i, N):
                k += self.K[i][j]
        
        return k  

    
    def set_initial_config(self):
        """
        Positions spring nodes on sites of a triangular lattice, which is 
        subsequently wrapped around a cylinder (i.e., x -> z and y -> phi). 
        The cylinder dimensions are determined such that the nearest neighbor 
        geodesic distance = 1.
        """
        a = np.zeros((2,2))
        if self.orient == 0:  # for compression in [11] crystal direction
            a = np.array([[1,0],[0.5,np.sqrt(3)/2]])
        elif self.orient == 1:  # for compression in [10] crystal direction
            a = np.array([[np.sqrt(3),0],[np.sqrt(3)/2,0.5]])
        else:
            print('invalid orientation')    
            
        self.cylRad = self.cylRad_0 = self.Ny*a[1][1]/(2*np.pi)
        self.cylLen = self.cylLen_0 = (self.Nx-1)*a[0][0] + a[1][0]   
        
        Nx = self.Nx    
        Ny = self.Ny
        
        for i in range(Nx):
            for j in range(Ny):
                n = i + j*Nx  # node index convention throughout
                self.X[n][0] = i*a[0][0] + j*a[1][0]  # x-component
                self.X[n][1] = i*a[0][1] + j*a[1][1]  # y-component                               
                self.X[n][0] -= (j//2)*a[0][0] # make overall shape rectangular                
                self.R[n][0] = self.X[n][0]              # axial coordinate      
                self.R[n][1] = self.X[n][1]/self.cylRad  # angular coordinate
                
        self.R_0 = self.R
        
        
    def get_neighbor_list(self, m):
        """
        Returns a 1d array containing indices of node m's nearest neighbors. 
        The array has length 6, and if the local coordination number < 6, 
        array elements with negative values are to be regarded as non-entries.
        """
        Nx = self.Nx
        Ny = self.Ny
        N  = self.N
        j = m//Nx
        i = m - j*Nx  # consistent with indexing in set_initial_positions()
        nhbr = np.zeros(6)     
    
        if self.orient == 0:
            nhbr[0] = m-1 if i>0 else -1        # W nhbr
            nhbr[1] = m+1 if i<Nx-1 else -1     # E
            #####                               # NW    
            if j==Ny-1:
                nhbr[2] = m - (N-Nx)
            elif i>0 or j%2==1:
                nhbr[2] = m + Nx - 1 + j%2
            else:
                nhbr[2] = -1
            #####                               # NE
            if j==Ny-1 and i<Nx-1:
                nhbr[3] = m - (N-Nx) + 1
            elif i<Nx-1 or j%2==0:
                nhbr[3] = m + Nx + j%2
            else:
                nhbr[3] = -1
            #####                               # SW
            if j==0 and i>0:
                nhbr[4] = m + (N-Nx) - 1
            elif i>0 or j%2==1:
                nhbr[4] = m - Nx - 1 + j%2
            else:
                nhbr[4] = -1
            #####                               # SE
            if j==0:
                nhbr[5] = m + (N-Nx)
            elif i<Nx-1 or j%2==0:
                nhbr[5] = m - Nx + j%2
            else:
                nhbr[5] = -1
            
        elif self.orient == 1:                                             
            nhbr[0] = m+2*Nx if j<Ny-2 else m-(N-2*Nx)  # N nhbr                                       
            nhbr[1] = m-2*Nx if j>1 else m+(N-2*Nx)     # S 
            #####                                       # NW
            if j==Ny-1:
                nhbr[2] = m - (N-Nx)
            elif i>0 or j%2==1:
                nhbr[2] = m + Nx - 1 + j%2
            else:
                nhbr[2] = -1
            #####                                       # NE
            if j==Ny-1 and m<N-1:
                nhbr[3] = m - (N-Nx) + 1
            elif i<Nx-1 or j%2==0:
                nhbr[3] = m + Nx + j%2
            else:
                nhbr[3] = -1                    
            #####                                       # SW
            if j==0 and m>0:
                nhbr[4] = m + (N-Nx) - 1
            elif i>0 or j%2==1:
                nhbr[4] = m - Nx - 1 + j%2
            else:
                nhbr[4] = -1
            #####                                       # SE
            if j==0:
                nhbr[5] = m + (N-Nx)
            elif i<Nx-1 or j%2==0:
                nhbr[5] = m - Nx + j%2
            else:
                nhbr[5] = -1
        
        else:
            print('invalid lattice') 
                          
        return nhbr.astype(int) 
    

    def draw_network(self, k, success, dots=True):
        """
        Puts a dot at the location of each node (if dots=True), and draws a 
        line segment representing each spring. For simplicity, the springs 
        connecting the system to its periodic copies are omitted from the 
        drawing. May be nice to put them in later... 
        """
        plt.figure() # make new plot window       
        Xt = self.X.T  # transposed matrix of node positions
               
        if dots:
            plt.scatter(Xt[0], Xt[1])
                
        axes = plt.gca()
        axes.set_xlim([np.amin(Xt[0])-1, np.amax(Xt[0])+1])
        axes.set_ylim([np.amin(Xt[1])-1, np.amax(Xt[1])+1]) 
        axes.set_aspect('equal')
        
        # that was the node part, now do the springs
        lineSegments = []
        for i in range(self.N):
            nhbr = self.get_neighbor_list(i)
            for n in range(6):
                j = nhbr[n]
                if j > i: # avoids double-counting and non-neighbors (j<0)
                    dy = np.abs(self.X[i][1] - self.X[j][1])
                    short = (dy < np.pi*self.cylRad)
                    if short:
                        if self.K[i][j] > 0:
                            lineSegments.append([i, j])
        
        lines = [[tuple(self.X[j]) for j in i]for i in lineSegments]
        lc = col.LineCollection(lines)
        axes.add_collection(lc)
        
        #drawing filled inclusions
        X = self.X         
        for i in range(self.num_of_incl):
            vertices = self.triangle_vertices(self.position[i],
                                              self.orientation[i])
            
            x=np.array([X[np.int(vertices[0])][0],X[np.int(vertices[1])][0],X[np.int(vertices[2])][0]])
            y=np.array([X[np.int(vertices[0])][1],X[np.int(vertices[1])][1],X[np.int(vertices[2])][1]])
            axes.fill(x,y,"#B22222")
            
        r = self.get_coords()
        axes.set_title("Strain = {:g}% Energy = {:g} Converge = {}"
                       .format(self.strain*100, self.energy(r), success))       
        
        filename = "movie_frames/{:g}".format(np.int(k))
        plt.savefig(filename, dpi=400, bbox_inches='tight')
        plt.show()             
        
    def set_springs(self, K_hookean, K_sfp, K_cross, K_area, p):
        """
        Assigns nonzero elements to the matrix of spring constants. This method
        supports putting springs between all nearest-neighbor pairs and taking
        the spring constants from a uniform random distribution of specified 
        mean and width.
        """
        
        #np.random.seed(datetime.now())
        N = self.N
        nhbr = np.zeros(6)   
        Nx = self.Nx       
        self.K_hookean = K_hookean
        self.K_sfp = K_sfp
        self.K_cross = K_cross
        self.K_area = K_area
        
        #dilution
        for i in range(Nx, N):
            nhbr = self.get_neighbor_list(i)
            for j in range(i+1, N):
                if self.K[i][j] == 0:                        
                    if j==nhbr[0] or j==nhbr[1] or j==nhbr[2]\
                    or j==nhbr[3] or j==nhbr[4] or j==nhbr[5]:
                        if np.random.random() < p:
                            if i>=2*Nx and i<=3*Nx and j==nhbr[0]:
                                h = K_hookean
                            else:
                                h = K_hookean
                        else:
                            h = 0
                        self.K[i][j] = h
                        #self.K[i][j] = mean + width*(np.random.random() - 0.5)
                        self.K[j][i] = self.K[i][j]  # symmetrize
        #To create a hole near the center of the lattice of some size modify accordingly 
        for i in range(18,20):
            nhbr = self.get_neighbor_list(i)
            for j in range(i+1, N-Nx):
                if self.K[i][j] == 0 or self.K[i][j]!=0:                        
                    if j==nhbr[0] or j==nhbr[1] or j==nhbr[2]\
                    or j==nhbr[3] or j==nhbr[4] or j==nhbr[5]:
                        self.K[i][j] = 0.0
                        self.K[j][i] = self.K[i][j]  # symmetrize                
       
        for i in range(26,28):
            nhbr = self.get_neighbor_list(i)
            for j in range(i+1, N-Nx):
                if self.K[i][j] == 0 or self.K[i][j]!=0:                        
                    if j==nhbr[0] or j==nhbr[1] or j==nhbr[2]\
                    or j==nhbr[3] or j==nhbr[4] or j==nhbr[5]:
                        self.K[i][j] = 0.0
                        self.K[j][i] = self.K[i][j]  # symmetrize                
              

                        
        #To remove periodic boundary condition springs
        for i in range(Nx):
            nhbr = self.get_neighbor_list(i)
            for j in range(i+1, N-Nx):
                if self.K[i][j] == 0:                        
                    if j==nhbr[0] or j==nhbr[1] or j==nhbr[2]\
                    or j==nhbr[3] or j==nhbr[4] or j==nhbr[5]:
                        self.K[i][j] = self.random(p)
                        self.K[j][i] = self.K[i][j]  # symmetrize

    
    def random(self, p):
        out = 0.0
        if np.random.random() < p:
           out = self.K_hookean
        else:
            out = 0
            
        return out                     


    def energy(self, r):
        hookeanEnergy    = self.hookean_energy(r)
        sfpEnergy        = self.sfp_energy(r)
        crosslinkEnergy  = self.cross_link_energy(r)
        inclusionEnergy  = self.inclusion_energy(r)
        return hookeanEnergy + sfpEnergy + crosslinkEnergy + inclusionEnergy
    
    
    def hookean_energy(self, r):
        """
        To be passed as a bound method to an optimizer. r is a 1d array of the 
        DOFs. These are the z-coordinates of all but Ny/2 left edge nodes 
        followed by the phi-coordinates of all N nodes. Thus r should have 
        size 2*N-Ny/2. The radial coordinate is controlled by the 
        apply_compression() method and is taken as a parameter, not a variable.
        """              
        N = self.N
        Nx = self.Nx
        Rnew = np.zeros((N,2)) # to hold all current position data
        k = 0  # count
        for i in range(N):
            if i%(2*Nx)==0:
                Rnew[i][0] = self.R[i][0]  # z-coords of left edge nodes
            else:
                Rnew[i][0] = r[k]          # z-coords of all other nodes
                k += 1
        # k = N-Ny/2
        for i in range(N):
            Rnew[i][1] = r[i+k]            # phi-coords of all nodes
                   
        energy = 0.0
        for i in range(N):
            nhbr = self.get_neighbor_list(i)
            for n in range(6): 
                j = nhbr[n]
                if j > i: # avoids double-counting and non-neighbors (j<0)
                    dz   = Rnew[i][0] - Rnew[j][0] 
                    dphi = Rnew[i][1] - Rnew[j][1]
                    dphi = np.arcsin( np.sin(dphi) )
                    ds = self.cylRad*dphi  
                    rij = np.sqrt(dz**2 + ds**2) # geodesic distance
                    energy += 0.5*self.K[i][j]*(rij-1.0)**2    
        
        return energy 

    
    def sfp_energy(self, r):
        """
        #Puts three angular spring of rest angle pi around each vertex
        """
    
        N = self.N
        Nx = self.Nx
        K = self.K
        Rnew = self.R
        X = np.zeros((N,2))
        X_edges = np.zeros((N,2))
        
        k = 0  # count
        for i in range(N):
            if i%(2*Nx)==0:
                Rnew[i][0] = self.R[i][0]  # z-coords of left edge nodes
            else:
                Rnew[i][0] = r[k]          # z-coords of all other nodes
                k += 1
        # k = N-Ny/2
        for i in range(N):
            Rnew[i][1] = r[i+k]            # phi-coords of all nodes
            
        #cartesian coords
        for i in range(N):
            X[i][0] = Rnew[i][0]
            X[i][1] = Rnew[i][1]*self.cylRad            
        
        energy = 0        
        for o in range(Nx,N-Nx): #angular springs for all rows except top 
                                #and bottom
            nhbr = self.get_neighbor_list(o)
            #west <-> east
            l = nhbr[0]
            ri = nhbr[1]
            if l >= 0 and ri >= 0:     #check for boundary conditions
                if K[o][l]*K[o][ri] > 0: 
                                    #angular spring only if the entire line -
                                    #i.e.left and right edges of line are present
                    energy += (self.angle(X[l],X[o],X[ri]))**2                                    
            #north west <-> south east
            l = nhbr[2]
            ri = nhbr[5]
            if l >= 0 and ri >= 0:     
                if K[o][l]*K[o][ri] > 0: 
                    energy += (self.angle(X[l],X[o],X[ri]))**2    
            #south west <-> north east        
            l = nhbr[4]
            ri = nhbr[3]
            if l >= 0 and ri >= 0:     
                if K[o][l]*K[o][ri] > 0: 
                    energy += (self.angle(X[l],X[o],X[ri]))**2 
                    
        #putting angular springs for top and bottom row        
        for j in range(0,N,N-Nx):
            for i in range(N):
                X_edges[i][0] = np.copy(Rnew[i][0])
                X_edges[i][1] = np.copy(Rnew[i][1]*self.cylRad)   
            if j==0:            #bottom row
                for i in range(Nx):
                    X_edges[N-i-1][1] = X_edges[N-i-1][1] - 2*np.pi*self.cylRad
            elif j==N-Nx:          #top row
                for i in range(Nx):
                    X_edges[i][1] = X_edges[i][1] + 2*np.pi*self.cylRad 
                    self.X_edges=X_edges                               
            for o in range(Nx): #j=0 angular spring energies for bottom row
                nhbr = self.get_neighbor_list(o+j)
                #west <-> east
                l = nhbr[0]
                ri = nhbr[1]
                if l >= 0 and ri >= 0:     
                    if K[o+j][l]*K[o+j][ri] > 0:                                     
                        energy += (self.angle(X_edges[l],X_edges[o+j],X_edges[ri]))**2                                    
                #north west <-> south east
                l = nhbr[2]
                ri = nhbr[5]
                if l >= 0 and ri >= 0:     
                    if K[o+j][l]*K[o+j][ri] > 0: 
                        energy += (self.angle(X_edges[l],X_edges[o+j],X_edges[ri]))**2    
                #south west <-> north east        
                l = nhbr[4]
                ri = nhbr[3]
                if l >= 0 and ri >= 0:     
                    if K[o+j][l]*K[o+j][ri] > 0: 
                        energy += (self.angle(X_edges[l],X_edges[o+j],X_edges[ri]))**2      
             
        return (self.K_sfp/2)*energy         
            
        
    def cross_link_energy(self, r):
        """
        Puts three angular spring of rest angle pi around each vertex
        """
    
        N = self.N
        Nx = self.Nx
        K = self.K
        Rnew = self.R
        X = np.zeros((N,2))
        
        k = 0  # count
        for i in range(N):
            if i%(2*Nx)==0:
                Rnew[i][0] = self.R[i][0]  # z-coords of left edge nodes
            else:
                Rnew[i][0] = r[k]          # z-coords of all other nodes
                k += 1
        # k = N-Ny/2
        for i in range(N):
            Rnew[i][1] = r[i+k]            # phi-coords of all nodes
            
        #cartesian coords
        for i in range(N):
            X[i][0] = Rnew[i][0]
            X[i][1] = Rnew[i][1]*self.cylRad            
        
        energy = 0            
        for i in range(N):
            nhbr = self.get_neighbor_list(i)
            nhbr = self.rearrange_nhbr(nhbr) #arranges nhbr list in cyclic order
            
            for j in range(6):
                if nhbr[np.mod(j-1,6)]>=0 and nhbr[j] >=0:
                    l = np.int(nhbr[np.mod(j-1,6)])
                    ri = np.int(nhbr[j])
                    if K[i][l]*K[i][ri] > 0: 
                        #cross-link only when both springs are present
                        theta = self.angle(X[l], X[i], X[ri])
                        energy += (theta+np.pi/3)**2 
                        #theta output is -ve pi/3 degree at zero strain
  
        return self.K_cross/2*energy         
    
    
    def angle(self, l, o, ri):      
        """
        calculates angle made by three vertices.
        left vertex, center vertex, right vertex (arrays passed in this order)
        """        
                
        left  = l - o       #left edge vector
        right = ri - o      #right edge vector        
        
        cross = left[0]*right[1] - left[1]*right[0]
        angle = np.arcsin(cross/(np.sqrt(left.dot(left))*
                                 np.sqrt(right.dot(right))))
        
        return angle    


    def rearrange_nhbr(self,nhbr):
        """
        arranges nhbr list in cyclic order
        """
        nhbr2 = np.zeros(6)
        nhbr2[0] = nhbr[3]
        nhbr2[1] = nhbr[1]
        nhbr2[2] = nhbr[5]
        nhbr2[3] = nhbr[4]
        nhbr2[4] = nhbr[0]
        nhbr2[5] = nhbr[2]           
        
        return nhbr2
    
    
    def apply_compression(self, epsilon):
        """
        Reduces the radius of the cylinder to R0*(1-epsilon), where R0 is the 
        original radius, and epsilon is the compressive strain (defined > 0).
        Note this method only rescales the y-coordinates; a subsequent call to 
        an optimizer should thus be made.
        """
        self.strain = epsilon
        self.cylRad = self.cylRad_0*(1-epsilon)
        for i in range(self.N):
            self.X[i][1] = self.R[i][1]*self.cylRad  # rescale y-coordinates
                         
        
    def get_coords(self):
        """
        Returns all but left edge nodes' z-coordinates and all nodes' phi-
        coordinates as a 1d array.
        """
        N = self.N
        Nx = self.Nx
        Ny = self.Ny
        r = np.zeros(2*N-Ny//2) # to hold all current position data
        k = 0  # count
        for i in range(N):
            if i%(2*Nx) != 0:
                r[k] = self.R[i][0]
                k += 1
        # k = N-Ny/2
        for i in range(N):
            r[i+k] = self.R[i][1]
        return r
   

    def set_coords(self, r):
        """
        Sets all but left edge nodes' z-coordinates and all nodes' phi-
        coordinates to the 1d array of supplied values. The cylinder length
        also gets updated by this method.
        """
        N = self.N
        Nx = self.Nx
        k = 0  # count
        for i in range(N):
            if i%(2*Nx) != 0:
                self.R[i][0] = r[k]
                k += 1
        # k = N-Ny/2
        for i in range(N):
            self.R[i][1] = r[i+k]
            
        # set cylinder length equal to the right edge nodes' AVERAGE z value
        zvals, phivals = self.get_edge()
        self.cylLen = np.mean(zvals)
        
        # update cartesian coords
        for i in range(N):
            self.X[i][0] = self.R[i][0]
            self.X[i][1] = self.R[i][1]*self.cylRad
                

    def get_edge(self):
        """
        Returns two arrays of size Ny/2 containing the axial and angular
        coordinates of the free edge nodes.
        """
        Nx = self.Nx
        Ny = self.Ny
        Nedge = Ny//2
        zvals   = np.zeros(Nedge)
        phivals = np.zeros(Nedge)
        for i in range(Nedge):
            n = 2*(i+1)*Nx - 1
            zvals[i]   = self.R[n][0]
            phivals[i] = self.R[n][1]
        return zvals, phivals
    
        
    def set_inclusions(self, num_of_incl):
        self.num_of_incl = num_of_incl
        self.position = np.array([num_of_incl])
        self.orientation = np.array([num_of_incl])
        Nx = self.Nx
        Ny = self.Ny
        
        list_choose = []
        for i in range(1,Ny-1):
            a = np.arange(i*Nx+1,(i+1)*Nx-1)
            list_choose = np.append(list_choose, a)
        
        num_choices = len(list_choose)
        #self.position = np.random.randint(0,num_choices,size=num_of_incl)
        self.position = np.random.choice(num_choices, num_of_incl, replace=False)
        for i in range(num_of_incl):
            self.position[i] = list_choose[np.int(self.position[i])]
            
        #self.position = self.Nx + np.random.randint(0,self.N-2*self.Nx,size=num_of_incl)
        #returns a random position and excludes top and bottom row
        #(since these two rows cannot have an inclusion in all orientations)
        self.orientation = np.random.randint(0,6,size=num_of_incl)
        #returns random integers within interval 0 to 5
                            
    def inclusion_energy(self, r):
        energy = 0
        for i in range(self.num_of_incl):
            area = self.calculate_area(r, self.position[i], self.orientation[i])
            energy += (area - self.area0)**2  
        
        return self.K_area/2*energy
    
    
    def calculate_area(self, r, position, orientation):        
        vertices = self.triangle_vertices(position, orientation)
        
        N = self.N
        Nx = self.Nx
        Rnew = np.zeros((N,2)) # to hold all current position data
        k = 0  # count
        for i in range(N):
            if i%(2*Nx)==0:
                Rnew[i][0] = self.R[i][0]  # z-coords of left edge nodes
            else:
                Rnew[i][0] = r[k]          # z-coords of all other nodes
                k += 1
        # k = N-Ny/2
        for i in range(N):
            Rnew[i][1] = r[i+k]            # phi-coords of all nodes
        
        xy = np.zeros([3,2])
        for i in range(3):
            xy[i][0] = self.cylRad*Rnew[np.int(vertices[i])][0] 
            xy[i][1] = Rnew[np.int(vertices[i])][1]          
                        
        area = 0
        for i in range(3):
            area += xy[i][0]*xy[np.mod(i+1,3)][1] - xy[np.mod(i+1,3)][0]*xy[i][1]
        area  = abs(area)/2        
        
        return area
    
    
    def triangle_vertices(self, position, orientation):
        nhbr = self.get_neighbor_list(position)
        neighbors = np.array([nhbr[2],nhbr[3],nhbr[1],nhbr[5],nhbr[4],nhbr[0]])
        #clockwise arrangement of vertices around position, starting from NW
        
        vertices = np.zeros(3)
        vertices[0] = position
        vertices[1] = neighbors[orientation]
        vertices[2] = neighbors[np.mod(orientation+1, 6)]
        
        return vertices

                       
    def store(self, runs, maxStrain, Nsteps, h, energy_runs, sfp_energy_runs, 
              crosslink_energy_runs, hookean_energy_runs, inclusion_energy_runs,
              success_runs, parameters):
              
        deltaStrain = maxStrain/Nsteps
        strain = 100*np.arange(deltaStrain, maxStrain+deltaStrain, deltaStrain)
        
        energy_average = np.sum(energy_runs, 0)/ runs    
        energy_std_dev = np.std(energy_runs, axis=0)
        
        z = np.array([strain, energy_average, energy_std_dev])
        np.savetxt("total_energy{0:.2f}.dat".format(h), z.T, header = parameters)
        
        stress = np.zeros(Nsteps)
        for i in range(Nsteps - 1):
            stress[i] = energy_average[i+1] - energy_average[i]
        stress = stress/deltaStrain    
        z = np.array([strain, stress])
        np.savetxt("stress{0:.2f}.dat".format(h), z.T, header = parameters)
        
        stiffness = np.zeros(Nsteps)
        for i in range(Nsteps - 2):
            stiffness[i] = energy_average[i+2] - 2*energy_average[i+1] + energy_average[i]
        stiffness = stiffness/deltaStrain**2    
        z = np.array([strain, stiffness])
        np.savetxt("stiffness{0:.2f}.dat".format(h), z.T, header = parameters)      
        
        sfp_energy_average      = np.sum(sfp_energy_runs, 0)/ runs    
        sfp_energy_std_dev      = np.std(sfp_energy_runs, axis=0)
        
        crosslink_average= np.sum(crosslink_energy_runs, 0)/runs
        crosslink_std_dev= np.std(crosslink_energy_runs, axis=0)
        
        hook_energy_average     = np.sum(hookean_energy_runs, 0)/ runs    
        hook_energy_std_dev     = np.std(hookean_energy_runs, axis=0)
        
        inclusion_energy_average     = np.sum(inclusion_energy_runs, 0)/ runs    
        inclusion_energy_std_dev     = np.std(inclusion_energy_runs, axis=0)
                           
        z = np.array([strain, sfp_energy_average, sfp_energy_std_dev])
        np.savetxt("sfp_energy{0:.2f}.dat".format(h), z.T, header = parameters)
        
        z = np.array([strain, crosslink_average, crosslink_std_dev])
        np.savetxt("crosslink_energy{0:.2f}.dat".format(h), z.T, header = parameters)
        
        z = np.array([strain, hook_energy_average, hook_energy_std_dev])
        np.savetxt("hookean_energy{0:.2f}.dat".format(h), z.T, header = parameters)
        
        z = np.array([strain, inclusion_energy_average, inclusion_energy_std_dev])
        np.savetxt("inclusion_energy{0:.2f}.dat".format(h), z.T, header = parameters)
        
        success_average = np.sum(success_runs, 0)/ runs    
        success_std_dev = np.std(success_runs, axis=0)  
        
        z = np.array([strain, success_average, success_std_dev])
        np.savetxt("success{0:.2f}.dat".format(h), z.T, header = parameters)
        
        np.save("total_energy{0:.2f}".format(h), energy_runs)
        np.save("sfp_energy{0:.2f}".format(h), sfp_energy_runs)
        np.save("crosslink_energy{0:.2f}".format(h), crosslink_energy_runs)
        np.save("hookean_energy{0:.2f}".format(h), hookean_energy_runs)
        np.save("success{0:.2f}".format(h), success_runs)


    def bounds(self):
        """
        To put bounds on phi-coordinates of vertices
        works only for energy2
        """
        N  = self.N
        Ny = self.Ny
        a1 = np.array([[None, None]])  #no bounds for z-coordinates
        a2 = np.array([[0, 2*np.pi-2*np.pi/Ny]])
        
        bnds = a1
        
        for i in range(N-Ny//2-1):
            bnds = np.append(bnds, a1, axis = 0)
        
        for i in range(N):
            bnds = np.append(bnds, a2, axis = 0)
            
        bnds = tuple(map(tuple,bnds))

        return bnds    
        
        
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
