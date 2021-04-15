import cv2
import math 
import numpy as np


class Snake:


    ####################Constant Parameters####################
    min_distance_b_points = 5               # The minimum distance between two points to consider them overlaped
    max_distance_b_points = 50              # The maximum distance to insert another point into the spline
    kernel_size_search = 7                  # The size of search kernel


    ######################Variables#########################
    closed = True       # Indicates if the snake is closed or open.
    alpha = 0.5         # The weight of the uniformity energy.
    beta = 0.5          # The weight of the curvature energy.
    gamma = 0.5         # The weight of the Image (gradient) Energy.
    n_starting_points = 50       # The number of starting points of the snake.
    snake_length = 0
    #image = None        # The source image.
    gray = None         # The image in grayscale.
    binary = None       # The image in binary (threshold method).
    gradientX = None    # The gradient (sobel) of the image relative to x.
    gradientY = None    # The gradient (sobel) of the image relative to y.
    points = None

##############Define the constructor####################################
    def __init__(self,image,closed = True):
         # Sets the image and it's properties
        self.image = image

        # Image properties
        self.width = image.shape[1]
        self.height = image.shape[0]

        ######Set the line or curve to be closed loop ###################
        self.closed = closed

        # Image variations used by the snake
        self.gray = cv2.cvtColor( self.image, cv2.COLOR_RGB2GRAY )
        self.binary = cv2.adaptiveThreshold( self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2 )
        self.gradientX = cv2.Sobel( self.gray, cv2.CV_64F, 1, 0, ksize=5 )
        self.gradientY = cv2.Sobel( self.gray, cv2.CV_64F, 0, 1, ksize=5 )


        ######To Draw our circle we need to define the center of the image As follow
        half_width = math.floor( self.width / 2 )
        half_height = math.floor( self.height / 2 )

        ###########If we use the closed loop, we put the large circle that will reduce untill round the edges of objects######
        if self.closed:
            n = self.n_starting_points
            radius = half_width if half_width < half_height else half_height
            self.points = [ np.array([
                half_width + math.floor( math.cos( 2 * math.pi / n * x ) * radius ),
                half_height + math.floor( math.sin( 2 * math.pi / n * x ) * radius ) ])
                for x in range( 0, n )
            ]
        else:   # If it is an open snake, the initial guess will be an horizontal line
            n = self.n_starting_points
            factor = math.floor( half_width / (self.n_starting_points-1) )
            self.points = [ np.array([ math.floor( half_width / 2 ) + x * factor, half_height ])
                for x in range( 0, n )
            ]

    
    def visuaize_Image(self):
        img = self.image.copy()

        # To draw lines between points, we have to define some parameters
        point_color = ( 0, 0, 255 )     # BGR RED
        line_color = ( 128, 0, 0 )      # BGR half blue
        thickness = 2                   # Thickness of the lines and circles

        num_points = len(self.points)

        # Draw a line between the current and the next point
        for i in range( 0, num_points - 1 ):
            cv2.line( img, tuple( self.points[ i ] ), tuple( self.points[ i + 1 ] ), line_color, thickness )

        # 0 -> N (Closes the snake)
        if self.closed:
            cv2.line(img, tuple( self.points[ 0 ] ), tuple( self.points[ num_points-1 ] ), line_color, thickness )

        # Drawing circles over points
        [ cv2.circle( img, tuple( x ), thickness, point_color, -1) for x in self.points ]

        return img


    ### Distance function controls the insertion and removing points around object
    def dist (first_point,second_point):

        return np.sqrt(np.sum((first_point-second_point) ** 2))



    ####### Normalization funtion to normalize the kernel of search
    def normalize (kernel):
        # abs_sum = 0
        # for i in kernel:
        #     abs_sum += abs(i)

        # if abs_sum !=0:
        #     return kernel/abs_sum
        # else:
        #     return kernel
        abs_sum = np.sum( [ abs( x ) for x in kernel ] )
        return kernel / abs_sum if abs_sum != 0 else kernel
        

    def get_length(self):

        n_points = len(self.points)
        if not self.closed:
            n_points -= 1

        return np.sum( [ Snake.dist( self.points[i], self.points[ (i+1)%n_points  ] ) for i in range( 0, n_points ) ] )

#############Define the internal and external energy
# Continuity energy is the the external energy 
    def cont_energy(self, p, prev):

        snakey_len = self.snake_length
        points_len = len(self.points)
        # The average distance between points in the snake
        avg_dist = self.snake_length / len(self.points)
        # The distance between the previous and the point being analysed
        un = Snake.dist( prev, p )

        dun = abs( un - avg_dist )

        return dun**2

# Curveture energy
    def curv_energy(self, p, prev, next ):
        ### P : refer to the current point we want to get this energy
        #### prev : refer to the previous point we gonna to get there energy
        ##### next: refer to the next point we gonna to get there energy

        #first the distance between the previous point and the current point
        dis_x = p[0] - prev[0]
        dis_y = p[1] - prev[1]
        distant = math.sqrt(dis_x**2 + dis_y**2)

        #Second: The distance between the currrent and next points
        vx = p[0] - next[0]
        vy = p[1] - next[1]
        vn = math.sqrt( vx**2 + vy**2 )

        if distant == 0 or vn == 0:
            return 0

        cx = float( vx + dis_x )  / ( distant * vn )
        cy = float( vy + dis_y ) / ( distant * vn )
        cn = cx**2 + cy**2
        return cn


# Image (Gradient) Energy
    def Grad_energy(self,p):
        if p[0] < 0 or p[0] >= self.width or p[1] < 0 or p[1] >= self.height:
            # we use finfo function to get the objects are cached
            return np.finfo(np.float64).max

        return -( self.gradientX[ p[1] ][ p[0] ]**2 + self.gradientY[ p[1] ][ p[0] ]**2  )


    def set_alpha(self,parm):
        self.alpha = parm /100

    def set_beta(self,parm):
        self.beta = parm/100

    def set_gamma(self,parm):
        self.gamma = parm/100

    def remove_overlaping_points(self):
        
        size_points = len(self.points)

        for i in range(0,size_points+1,1):
            for j in range(size_points-1,i+1,-1):
                if i==j:
                    continue
        
                current = self.points[ i ]
                end = self.points[ j ]

                dist = Snake.dist( current, end )

                if dist < self.min_distance_b_points:
                    remove_indexes = range( i+1, j ) if (i!=0 and j!=snake_size-1) else [j]
                    remove_size = len( remove_indexes )
                    non_remove_size = snake_size - remove_size
                    if non_remove_size > remove_size:
                        self.points = [ p for k,p in enumerate( self.points ) if k not in remove_indexes ]
                    else:
                        self.points = [ p for k,p in enumerate( self.points ) if k in remove_indexes ]
                    snake_size = len( self.points )
                    break


    def add_missing_points(self):
        snake_size = len( self.points )
        for i in range(0,snake_size):
            curr = self.points[i]
            prev = self.points[(i+snake_size-1)%snake_size]
            next = self.points[(i+1)%snake_size]
            next2 = self.points[(i+2)%snake_size]

            if Snake.dist(curr,next) > self.max_distance_b_points:
                c0 = 0.125 / 6.0
                c1 = 2.875 / 6.0
                c2 = 2.875 / 6.0
                c3 = 0.125 / 6.0
                x = prev[0] * c3 + curr[0] * c2 + next[0] * c1 + next2[0] * c0
                y = prev[1] * c3 + curr[1] * c2 + next[1] * c1 + next2[1] * c0

                new_point = np.array( [ math.floor( 0.5 + x ), math.floor( 0.5 + y ) ] )

                self.points.insert( i+1, new_point )
                snake_size += 1


    def step(self):
        changed = False
        self.snake_length = self.get_length()
        new_snake = self.points.copy()

        search_kernel_size = (self.kernel_size_search,self.kernel_size_search)
        hks = math.floor(self.kernel_size_search/2)
        energy_cont = np.zeros(search_kernel_size)
        energy_curv = np.zeros(search_kernel_size)
        energy_grad = np.zeros(search_kernel_size)

        for i in range( 0, len( self.points ) ):
            curr = self.points[ i ]
            prev = self.points[ ( i + len( self.points )-1 ) % len( self.points ) ]
            next = self.points[ ( i + 1 ) % len( self.points ) ]

            for dx in range( -hks, hks ):
                for dy in range( -hks, hks ):
                    p = np.array( [curr[0] + dx, curr[1] + dy] )

                    # Calculates the energy functions on p
                    energy_cont[ dx + hks ][ dy + hks ] = self.cont_energy(p,prev)
                    energy_curv[dx+hks][dy+hks] = self.curv_energy(p,prev,next)
                    energy_grad[dx + hks][dy + hks] = self.Grad_energy(p)


            #Then, normalize the energies
            energy_cont = Snake.normalize(energy_cont)
            energy_curv = Snake.normalize(energy_curv)
            energy_grad = Snake.normalize(energy_grad)

            e_sum = self.alpha * energy_cont + self.beta * energy_curv + self.gamma * energy_grad
            emin = np.finfo(np.float64).max

            x,y = 0,0
            for dx in range( -hks, hks ):
                for dy in range( -hks, hks ):
                    if e_sum[ dx + hks ][ dy + hks ] < emin:
                        emin = e_sum[ dx + hks ][ dy + hks ]
                        x = curr[0] + dx
                        y = curr[1] + dy
            
            # Boundary check
            x = 1 if x < 1 else x
            x = self.width-2 if x >= self.width-1 else x
            y = 1 if y < 1 else y
            y = self.height-2 if y >= self.height-1 else y

            # Check for changes
            if curr[0] != x or curr[1] != y:
                changed = True

            new_snake[i] = np.array( [ x, y ] )

        self.points = new_snake

        # Post threatment to the snake, remove overlaping points and
        # add missing points
        self.remove_overlaping_points()
        self.add_missing_points()

        return changed









