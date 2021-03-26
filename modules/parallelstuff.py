# -*- coding: utf-8 -*-

# https://stackoverflow.com/questions/3033952/threading-pool-similar-to-the-multiprocessing-pool#3386632

recovery_sleep_time_seconds = 1 # number of seconds to wait before trying to launch another batch again

import numpy as np
import multiprocessing as mp
import copy
import time

# handle 'IOError: [Errno 4] Interrupted system call' errors from multiprocessing.Queue.get
#https://stackoverflow.com/questions/14136195/what-is-the-proper-way-to-handle-in-python-ioerror-errno-4-interrupted-syst
import errno
def my_queue_get(queue, block=True, timeout=None):
    while True:
        try:
            return queue.get(block, timeout)
        except IOError as e:
            if e.errno != errno.EINTR:
                raise
# Now replace instances of queue.get() with my_queue_get(queue), with other
# parameters passed as usual.
                    
# see here https://eli.thegreenplace.net/2012/01/16/python-parallelizing-cpu-bound-tasks-with-multiprocessing/
# and here https://stackoverflow.com/questions/37060091/multiprocessing-inside-function
try:
    
    from scipy.optimize import minimize
    
    def mworker(f,x0,fargs,margs,out_q):
        # worker invoked in a process puts the results in the output queue out_q
    #    f,x0,fargs,margs = args
        #print 'worker: fargs = ',fargs
        #print 'worker: margs = ',margs
        res = minimize(f, x0, args = fargs, **margs)
        #return [res.x, res.fun]
        out_q.put([[res.x, res.fun[0][0]]])

    # parallelize minimizations using different starting positions using multiprocessing, scipy.optimize.minimize
    def parallelminimize(f,x0s,fargs,margs,v0best=None,relative_bounds=None):
        # f is fcn to minimize
        # x0s are positions to start search from
        # fargs are arguments to pass to f
        # margs are arguments to pass to scipy.optimize.minimize
        
        # arguments to loop over
        if type(relative_bounds) is not type(None): # static bounds
            args = [(f,x,fargs,margs) for x in x0s]
        else: # relative bounds
            args = []
            for x in x0s:
                thesemargs = copy.copy(margs)
                thesemargs['bounds'] = (x + relative_bounds.T).T # it works. deal with it.
                args += [(f,x,fargs,thesemargs)]
        #print '\n[arg[3][\'bounds\'] for arg in args] = \n', [arg[3]['bounds'] for arg in args], '\n'
        
        ## add relative bounds
        #if type(relative_bounds) is not type(None):
            #for i in range(len(args)):
                #args[i][3]['bounds'] = args[i][1] + relative_bounds
            #print '\n[arg[3][\'bounds\'] for arg in args] = \n', [arg[3]['bounds'] for arg in args], '\n'
            ##print '\nargs = ', args, '\n'

        # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop#9786225
        # also could try concurrent futures
    #    import multiprocessing
    #    pool = multiprocessing.Pool()
    #    res = np.array(pool.map(minimizeone, args))
    #    res = np.array(pool.map(l, range(10)))
        
        # seems like this maybe be needed 
        # https://stackoverflow.com/questions/37060091/multiprocessing-inside-function
                
        worker = mworker
        nrun = len(x0s)
        nprocs = int(mp.cpu_count())# len(x0s)
        nbatch = int(np.floor(nrun / nprocs))
        if nrun % nprocs:
            nbatch += 1
        res = []
        
        # Each process will get a queue to put its result in it
        queues = [mp.Queue() for p in range(nprocs)]

        for b in range(nbatch):

            # try running a batch until it works
            while True:
                try:
                    procs = []
                    
                    ilow = b*nprocs
                    ihigh = min(nrun,(b+1)*nprocs)
                        
                    #print 'launching processes'

                    for i in range(ilow, ihigh):
                        p = mp.Process(
                                target=worker,
                                args=args[i]+tuple([queues[i-ilow]]))
                        procs.append(p)
                        p.start()
                        
                    #print 'collecting results'

                    for i in range(ilow, ihigh):
                        res += my_queue_get(queues[i-ilow]) # grab from this queue
                        
                    #print 'waiting for termination/cleanup'

                    # waits for worker to finish
                    for p in procs:
                        p.join()
                        p.terminate() # send SIGTERM just in case
                        del p # remove the multiprocessing.Process
                        
                    #print 'done with batch ', b+1, ' of ', nbatch
                    
                    break # made it this far so break out of the while loop
                
                except:
                    time.sleep(recovery_sleep_time_seconds) # wait a bit for processes to close before trying again
            
        res = np.array(res)
        #print 'res = ', res
        res = res[res[:,1]==np.min(res[:,-1])][0]
        #print 'res = ', res
        #print 'selected min is ',res[-1]
        #res = np.array(res[0])
        #print 'res = ',res
        
        # check if there's a better point
        #print 'v0best = ', v0best
        if v0best is None:
            res = np.array(res[0])
        else:
            if v0best[-1] < res[-1]:
                res = np.array(v0best[:-1])
            else:
                res = np.array(res[0])
                
        #print 'res = ',res

        return res
except:
    print('parallelstuff - WARNING: Could not load parallelminimize.')
    pass

def mapworker(f,x,fargs,out_q):
    # worker invoked in a process puts the results in the output queue out_q
    #print 'f = ',f,'\tx = ',x,'\tfargs = ',fargs,'\tf(x, *fargs) = ',f(x, *fargs)
    #out_q.put([[x,f(x, *fargs)]])
    out_q.put([[f(x, *fargs)]])

# yuno stock have python?!
def parallelmap(f,xs,fargs):
    # f is fcn to map to
    # xs is list of coords to eval
    # fargs is a tuple of common arguments to pass to f
    
    # arguments to loop over
    
    args = [(f,x,fargs) for x in xs]

    # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop#9786225
    
    # seems like this maybe be needed 
    # https://stackoverflow.com/questions/37060091/multiprocessing-inside-function
        
    worker = mapworker
    nrun = len(xs)
    nprocs = int(mp.cpu_count())# len(x0s)
    nbatch = int(np.floor(nrun / nprocs))
    if nrun % nprocs:
        nbatch += 1
    res = []
    
    # Each process will get a queue to put its result in it
    queues = [mp.Queue() for p in range(nprocs)]

    for b in range(nbatch):

        # try running a batch until it works
        while True:
            try:
                procs = []
                
                ilow = b*nprocs
                ihigh = min(nrun,(b+1)*nprocs)
                    
                #print 'launching processes'

                for i in range(ilow, ihigh):
                    p = mp.Process(
                            target=worker,
                            args=args[i]+tuple([queues[i-ilow]]))
                    procs.append(p)
                    p.start()
                    
                #print 'collecting results'

                for i in range(ilow, ihigh):
                    res += my_queue_get(queues[i-ilow]) # grab from this queue
                    
                #print 'waiting for termination/cleanup'

                # waits for worker to finish
                for p in procs:
                    p.join()
                    p.terminate() # send SIGTERM just in case
                    del p # remove the multiprocessing.Process
                    
                #print 'done with batch ', b+1, ' of ', nbatch
                    
                break # made it this far so break out of the while loop
            
            except:
                time.sleep(recovery_sleep_time_seconds) # wait a bit for processes to close before trying again
        
    # sort by argument order passed (seems presorted but just in case)
    #res = res[res[:,0].argsort()] # this doesn't work in genreal; must mix back in xs for slicing

    return res 

# #try testing parallelmap with this
def testparallelmap(njobs=10, sleepmax=10.e-3): # sleepmax is maximum random sleep time in seconds
    #from parallelstuff import *
    import time
    import numpy as np
    def f(x):
        time.sleep(sleepmax*np.random.rand()) # random delay to check sorting
        #print 'x = ', x # check that the returns are scrambled in time
        return x
    res = parallelmap(f,range(njobs),()) # is the result in ascending order?
    print(res)
    if res == [[x] for x in range(njobs)]:
        print('returned in order')
    else:
        print('result is out of order')

def map2worker(f,fargs,out_q):
    # worker invoked in a process puts the results in the output queue out_q
    out_q.put([[f(*fargs)]])

# yuno stock have python?!
def parallelmap2(f,fargslist,hostlist=None):
    # f is fcn to map to
    # fargs is a list of tuples of arguments to pass to f
    
    # check if we should run locally or remotely
    runLocalQ = True
    if type(hostlist) is not type(None): # None => local run
        runLocalQ = False
        try:
            nhosts = len(hostlist)
        except:
            nhosts = 0
    
    # arguments to loop over    
    args = [(f,fargs) for fargs in fargslist]

    # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop#9786225
    
    # seems like this maybe be needed 
    # https://stackoverflow.com/questions/37060091/multiprocessing-inside-function
        
    worker = map2worker
    nrun = len(fargslist)
    if runLocalQ:
        nprocs = int(mp.cpu_count())# len(x0s)
    else:
        nprocs = len(hostlist) # run remotely
    nbatch = int(np.floor(nrun / nprocs))
    if nrun % nprocs:
        nbatch += 1
    res = []
    
    # Each process will get a queue to put its result in it
    queues = [mp.Queue() for p in range(nprocs)]

    for b in range(nbatch):

        # try running a batch until it works
        while True:
            try:
                procs = []
                
                ilow = b*nprocs
                ihigh = min(nrun,(b+1)*nprocs)
                    
                #print 'launching processes'

                for i in range(ilow, ihigh):
                    if runLocalQ:
                        Args = args[i] # copy this one's args
                    else:
                        if nhosts:
                            Args = (f, args[i][1]+[hostlist[(i-ilow) % nhosts]])
                        else:
                            Args = (f, args[i][1]+[hostlist])
                    Args += tuple([queues[i-ilow]]) # assign a queue
                    p = mp.Process(target=worker,args=Args)
                    procs.append(p)
                    p.start()
                    
                #print 'collecting results'

                for i in range(ilow, ihigh):
                    res += my_queue_get(queues[i-ilow]) # grab from this queue
                    
                #print 'waiting for termination/cleanup'

                # waits for worker to finish
                for p in procs:
                    p.join()
                    p.terminate() # send SIGTERM just in case
                    del p # remove the multiprocessing.Process
                    
                #print 'done with batch ', b+1, ' of ', nbatch
                    
                break # made it this far so break out of the while loop
            
            except:
                time.sleep(recovery_sleep_time_seconds) # wait a bit for processes to close before trying again
        
    # sort by argument order passed (seems presorted but just in case)
    #res = res[res[:,0].argsort()] # this doesn't work in genreal; must mix back in xs for slicing

    return res 
    
try:
    
    from scipy.special import erfinv
    #from hammersley import hammersley
    from .chaospy_sequences import create_hammersley_samples
        
    def eworker(f,x,fargs,out_q):
        # worker invoked in a process puts the results in the output queue out_q
        res = f(x, *fargs)
        out_q.put(np.hstack((x, res[0][0])))

    # eval function over a range of initial points neval and return the nkeep lowest function evals
    def parallelgridsearch(f,x0,lengths,fargs,neval,nkeep):
        # f is fcn to minimize
        # x0 is center of the search
        # lengths is an array of length scales
        # fargs are arguments to pass to f
        # neval is the number of points to evaluate the function on
        # nkeep is the number of the neval points to keep
        
        if nkeep > neval: nkeep = neval
        
        # generate points to search
        ndim = len(lengths)
        #nevalpp = neval + 1
        #x0s = np.array([hammersley(i,ndim,nevalpp) for i in range(1,nevalpp)]) # hammersley uniform in all dims
        #x0s = np.vstack(parallelmap(hammersley, range(1,nevalpp), (ndim,nevalpp)))
        x0s = create_hammersley_samples(order=neval, dim=ndim).T
        x0s = np.sqrt(2)*erfinv(-1+2*x0s) # normal in all dimensions
        x0s = np.transpose(np.array(lengths,ndmin=2).T * x0s.T) # scale each dimension by it's lenghth scale
        x0s = x0s + x0 # shift to recenter

        # arguments to loop over
        # Each process will get a queue to put its result in it
        args = [(f,x,fargs) for x in x0s]

        # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop#9786225
        # also could try concurrent futures
    #    import multiprocessing
    #    pool = multiprocessing.Pool()
    #    res = np.array(pool.map(minimizeone, args))
    #    res = np.array(pool.map(l, range(10)))
        
        # seems like this maybe be needed 
        # https://stackoverflow.com/questions/37060091/multiprocessing-inside-function
            
        worker = eworker
        nrun = neval
        nprocs = int(mp.cpu_count())
        nbatch = int(np.floor(nrun / nprocs))
        if nrun % nprocs:
            nbatch += 1
        res = []

        # Each process will get a queue to put its result in it
        queues = [mp.Queue() for p in range(nprocs)]

        for b in range(nbatch):

            # try running a batch until it works
            while True:
                try:
                    procs = []
                    ilow = b*nprocs
                    ihigh = min(nrun,(b+1)*nprocs)
                    #print 'launching processes
                    print('two')

                    for i in range(ilow, ihigh):
                        p = mp.Process(target=worker, args=args[i]+tuple([queues[i-ilow]]))
                        procs.append(p)
                        p.start()

                    #print 'collecting results'
                        
                    if len(res):
                        for i in range(ilow, ihigh):
                            res = np.vstack((res,my_queue_get(queues[i-ilow])))
                    else:
                        res = np.array(my_queue_get(queues[0]))
                        for i in range(ilow+1, ihigh):
                            res = np.vstack((res,my_queue_get(queues[i-ilow])))
                        
                    #print 'waiting for termination/cleanup'

                    # waits for worker to finish
                    for p in procs:
                        p.join() # wait until close
                        p.terminate() # send SIGTERM just in case
                        del p # remove the multiprocessing.Process
                        
                    #print 'done with batch ', b+1, ' of ', nbatch

                    break # made it this far so break out of the while loop
                

                except Exception as e:
                    print(e)


        ## return nkeep smallest values
        #res = np.array(res)
        ##print 'res = ',res
        #resy = np.sort(res[:,-1])
        #res = res[res[:,-1]<=resy[nkeep-1]] # list of nkeep coords and function evals there
        
        # return nkeep smallest values
        # sort then cut
        res = np.array(res)
        res = res[res[:,-1].argsort()] # sort by last column
        res = res[res[:,-1]<=res[nkeep-1,-1]] # list of nkeep coords and function evals there
        
        #print 'res smallest = ', res

        #print 'resy = ',resy
        #print len(res),' sets of coords for smallest function evals: ', res

        return res # return coords and fcn evals
        #return res[:,:-1] # return just coords
        #return res[:,:-1], res[:,-1] # return just coords
    
except:
    print('parallelstuff - WARNING: Could not load parallelgridsearch.')
    pass
