class Kmeans:
    """ km = Kmeans( X, k= or centers=, ... )
        in: either initial centers= for kmeans
            or k= [nsample=] for kmeanssample
        out: km.centers, km.Xtocentre, km.distances
        iterator:
            for jcentre, J in km:
                clustercentre = centers[jcentre]
                J indexes e.g. X[J], classes[J]
    """
    def __init__( self, X, k=0, centers=None, nsample=0, **kwargs ):
        self.X = X
        if centers is None:
            self.centers, self.Xtocentre, self.distances = kmeanssample(
                X, k=k, nsample=nsample, **kwargs )
        else:
            self.centers, self.Xtocentre, self.distances = kmeans(
                X, centers, **kwargs )

    def __iter__(self):
        for jc in range(len(self.centers)):
            yield jc, (self.Xtocentre == jc)

#...............................................................................
if __name__ == "__main__":
    import random
    import sys
    from time import time

    N = 10000
    dim = 10
    ncluster = 10
    kmsample = 100  # 0: random centers, > 0: kmeanssample
    kmdelta = .001
    kmiter = 10
    metric = "cityblock"  # "chebyshev" = max, "cityblock" L1,  Lqmetric
    seed = 1

    exec( "\n".join( sys.argv[1:] ))  # run this.py N= ...
    np.set_printoptions( 1, threshold=200, edgeitems=5, suppress=True )
    np.random.seed(seed)
    random.seed(seed)

    print( "N %d  dim %d  ncluster %d  kmsample %d  metric %s" % (
        N, dim, ncluster, kmsample, metric) )
    X = np.random.exponential( size=(N,dim) )
        # cf scikits-learn datasets/
    t0 = time()
    if kmsample > 0:
        centers, xtoc, dist = kmeanssample( X, ncluster, nsample=kmsample,
            delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    else:
        randomcenters = randomsample( X, ncluster )
        centers, xtoc, dist = kmeans( X, randomcenters,
            delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    print( "%.0f msec" % ((time() - t0) * 1000) )

    # also ~/py/np/kmeans/test-kmeans.py
