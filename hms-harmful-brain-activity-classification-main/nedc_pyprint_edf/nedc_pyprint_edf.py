#!/usr/bin/env python
#
# file: /data/isip/data/nedc_pyprint_edf/nedc_pyprint_edf.py
#
# revision history:
#
# 20201220 (JP): first version
#
# This is a self-contained Python distribution that demonstrates
# the proper way to read a header and a signal from an EDF file.
#------------------------------------------------------------------------------

# import system modules
#
import os
import sys

# import nedc_modules
#
import nedc_debug_tools as ndt
import nedc_edf_tools as net
import nedc_file_tools as nft
import nedc_mont_tools as nmt

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# define default argument values
#
DEF_BSIZE = int(10)
DEF_FORMAT_FLOAT = "float"
DEF_FORMAT_SHORT = "short"
DEF_MODE = False

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

# declare a global debug object so we can use it in functions
#
dbgl = ndt.Dbgl()

# function: nedc_pystream_edf
#
# arguments:
#  fname: filename to be processed
#  montage: a montage object to be used for processing
#  bsize: the block size to be used for printing
#  format: print as floats or short ints
#  mode: ASCII is false, binary is true
#  fp: an open file pointer
#
# return: a boolean indicating status
#
def nedc_pystream_edf(fname, montage, bsize, format, mode, fp = sys.stdout):

    # declare local variables
    #
    edf = net.Edf()

    # display an informational message
    #
    if dbgl > ndt.BRIEF:
        fp.write("%s (line: %s) %s: streaming the signal (%s)\n" %
                 (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))

    # expand the filename (checking for environment variables)
    #
    ffile = nft.get_fullpath(fname)

    # read the unscaled Edf signal
    #
    if (format == DEF_FORMAT_SHORT):
        (h, isig) = edf.read_edf(ffile, False, True)
        if  isig == None:
            fp.write("Error: %s (line: %s) %s: %s\n" %
                     (__FILE__, ndt.__LINE__, ndt.__NAME__,
                      "error reading signal as short ints"))
            return False
    else:
        (h, isig) = edf.read_edf(ffile, True, True)
        if  isig == None:
            fp.write("Error: %s (line: %s) %s: %s\n" %
                     (__FILE__, ndt.__LINE__, ndt.__NAME__,
                      "error reading signal as floats"))
            return False

    # apply the montage
    #
    #
    mnt = nmt.Montage()
    osig = mnt.apply(isig, montage);
    if  osig == None:
        fp.write("Error: %s (line: %s) %s: %s\n" %
                 (__FILE__, ndt.__LINE__, ndt.__NAME__,
                  "error applying montage"))
        return False
        
    # case: ascii mode
    #
    if mode is False:
    
        # get the number of samples per channel
        #
        key = next(iter(osig))
        nchannels = len(osig)
        i = int(0);
        iframe = int(0);
        iend = len(osig[key])

        # display some useful signal information
        #
        if dbgl > ndt.NONE:
            fp.write("mode is ascii\n")
            fp.write("number of output channels = %d\n" % (nchannels))
            fp.write("number of samples per channel = %d\n" % (iend))

        # loop over all samples
        #
        while i < iend:

            # display some information to make the output more readable
            #
            if dbgl > ndt.NONE:

                # display frame information
                #
                fp.write("frame %5d: sample %8d to %8d\n" %
                         (iframe, i, i + bsize))

                # display montage labels
                #
                fp.write("%8s " % (nft.STRING_EMPTY))
                for key in osig:
                    fp.write("%10s " % (key))
                fp.write(nft.DELIM_NEWLINE)

            # display the sample values
            #
            for j in range(i, i + bsize):

                # make sure we don't exceed the signal length
                #
                if j < iend:
                    fp.write("%8d:" % (j))
                    for key in osig:

                        # format == DEF_FORMAT_FLOAT:
                        #  display floating point
                        #
                        if format == DEF_FORMAT_FLOAT:
                            fp.write("%10.4f " % (osig[key][j]))
                        else:
                            fp.write("%10d " % (int(osig[key][j])))
                    fp.write(nft.DELIM_NEWLINE)

                # increment counters
                #
                i += bsize
                iframe += int(1)

    # case: binary mode
    #
    else:

        # get the number of samples per channel
        #
        key = next(iter(osig))
        nchannels = len(osig)
        i = int(0);
        iframe = int(0);
        iend = len(osig[key])

        # display some useful signal information
        #
        if dbgl > ndt.NONE:
            fp.write("mode is binary\n")
            fp.write("number of output channels = %d\n" % (nchannels))
            fp.write("number of samples per channel = %d\n" % (iend))

        # do a vector based conversion of the data for speed
        #
        if format == DEF_FORMAT_SHORT:
            tmp = {}
            for key in osig:
                tmp[key] = np.clip(osig[key],
                                   DEF_SHORT_MINVAL, DEF_SHORT_MAXVAL)
                tmp[key].round()

        # loop over the samples
        #
        while i < iend:

            # display some information to make the output more readable
            #
            if dbgl > ndt.NONE:

                # display frame information
                #
                fp.write("frame %5d: sample %8d to %8d\n" %
                         (iframe, i, i + bsize))

                # display montage labels
                #
                fp.write("%8s " % (nft.STRING_EMPTY))
                for key in osig:
                    fp.write("%10s " % (key))
                fp.write(nft.DELIM_NEWLINE)

            # display the sample values: write one channel at a time
            #
            for key in osig:
                for j in range(i, i + bsize):

	            # make sure we don't exceed the signal length
	            #
                    if j < iend:

                        # fmt == DEF_FORMAT_FLOAT: write binary data as floats
                        #  note that the signal is a double. note also these
                        #  lines don't really fit in our 80-col format :(
                        #
                        if format == DEF_FORMAT_FLOAT:
                            sys.stdout.buffer.write(struct.pack('<f',
                                                                osig[key][j]))
                        else:
                            sys.stdout.buffer.write(struct.pack('<h',
                                                                int(tmp[key][j])))

            # increment counters
            #
            i += bsize
            iframe += int(1)

    # exit gracefully
    #
    return True

# function: main
#
def main(argv):

    # declare local variables
    #
    edf = net.Edf()

    # parse the command line:
    #  we keep it simple - two filenames are the inputs
    #
    edf_file = sys.argv[1]
    param_file = sys.argv[2]

    # display an informational message
    #
    print("edf   file: %s" % (edf_file))
    print("param file: %s" % (param_file))
    print("")
     
    # read and print the header
    #
    print("<---------- start of header ---------->")
    edf.print_header_from_file(edf_file, sys.stdout)
    print("<---------- end   of header ---------->")
    print("")

    # read and print the header
    #
    print("<---------- start of signal ---------->")
    mnt = nmt.Montage()
    montage = mnt.load(param_file)
    nedc_pystream_edf(edf_file, montage, 
		      DEF_BSIZE, DEF_FORMAT_FLOAT, DEF_MODE,
		      sys.stdout)
    print("<---------- end   of signal ---------->")

    # exit gracefully
    #
    return True

# begin gracefully
#
if __name__ == '__main__':
    main(sys.argv[0:])

#
# end of file
