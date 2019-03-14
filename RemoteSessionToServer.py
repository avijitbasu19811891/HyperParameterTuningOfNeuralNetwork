"""
   Code to uplaod any file etc to remote server
   Can be used to FTP the prediction results to remote Server
"""
import ftplib
import os

def uploadFileFTP(sourceFilePath, destinationDirectory, server, username, password):
    myFTP = ftplib.FTP(server, username, password)
    if destinationDirectory in [name for name, data in list(remote.mlsd())]:
        print "Destination Directory does not exist. Creating it first"
        myFTP.mkd(destinationDirectory)
    # Changing Working Directory
    myFTP.cwd(destinationDirectory)
    if os.path.isfile(sourceFilePath):
        fh = open(sourceFilePath, 'rb')
        myFTP.storbinary('STOR %s' % f, fh)
        fh.close()
    else:
        print "Source File does not exist"import ftplib
import os

def uploadFileFTP(sourceFilePath, destinationDirectory, server, username, password):
    myFTP = ftplib.FTP(server, username, password)
    if destinationDirectory in [name for name, data in list(remote.mlsd())]:
        print "Destination Directory does not exist. Creating it first"
        myFTP.mkd(destinationDirectory)
    # Changing Working Directory
    myFTP.cwd(destinationDirectory)
    if os.path.isfile(sourceFilePath):
        fh = open(sourceFilePath, 'rb')
        myFTP.storbinary('STOR %s' % f, fh)
        fh.close()
    else:
        print "Source File does not exist"