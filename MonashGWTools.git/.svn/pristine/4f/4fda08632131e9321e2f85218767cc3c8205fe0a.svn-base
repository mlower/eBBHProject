

import lalsimulation as lalsim
import lal
import numpy as np

def AntennaResponse(RA, DEC, psi, tc, ifo):

    ## returns the antenna response functions, F_+ and F_x, for a

    # RA, DEC
    # psi: polarisation angle
    # tc: coalescence time (GPS)
    # IFO: either 'H1', 'L1' or 'V1'

    tgps = lal.LIGOTimeGPS(tc)
    gmst = lal.GreenwichMeanSiderealTime(tgps)

    if ifo == 'H1':
        diff = lal.LALDetectorIndexLHODIFF
    elif ifo == 'L1':
        diff = lal.LALDetectorIndexLLODIFF
    elif ifo == 'V1':
        diff = lal.LALDetectorIndexVIRGODIFF
    else:
        raise ValueError('detector not recognized: ' + ifo)


    fplus, fcross = lal.ComputeDetAMResponse(lal.CachedDetectors[diff].response,\
                                             RA, DEC, psi, gmst)

    return fplus, fcross
    
