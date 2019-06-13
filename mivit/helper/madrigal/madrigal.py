# madrigal.py

import madrigalWeb.madrigalWeb
import datetime as dt
import numpy as np
import os


def identify_file(t,instrument_code,file_code, user):

    # initialize MadrigalData object and establish conection with Madrigal website
    test =  madrigalWeb.madrigalWeb.MadrigalData('http://cedar.openmadrigal.org/')

    # find experiments on day in question
    expList = test.getExperiments(instrument_code, t.year, t.month, t.day, 0, 0, 0, t.year, t.month, t.day, 23, 59, 59)
    ID = [exp.id for exp in expList if t>=dt.datetime(exp.startyear,exp.startmonth,exp.startday,exp.starthour,exp.startmin,exp.startsec) and t<dt.datetime(exp.endyear,exp.endmonth,exp.endday,exp.endhour,exp.endmin,exp.endsec)][0]
    # find files associated with experiment
    fileList = test.getExperimentFiles(ID)
    # find files of the correct type (correct file code)
    datafile = [file.name for file in fileList if file.kindat==file_code][0]

    # create file path by appending the Madarigal file name to the data directory
    filename = './TestDataSets/'+datafile.split('/')[-1]

    # if file does not exist, download it
    if not os.path.exists(filename):
        print('DOWNLOADING '+filename)
        test.downloadFile(datafile,filename, user['fullname'], user['email'], user['affiliation'], 'hdf5')

    return filename


def target_index(targtime,tstmparray):
    # convert targtime to unix timestamp
    targtstmp = (targtime-dt.datetime.utcfromtimestamp(0)).total_seconds()
    # find index of time in timearray that is closest to targtime
    t = np.argmin(np.abs(tstmparray-targtstmp))
    return t
