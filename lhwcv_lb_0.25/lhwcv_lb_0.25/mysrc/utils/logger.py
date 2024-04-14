import  os
import time

import numpy as np

def get_timestamp():
    return time.strftime('%Y-%m-%d_%H:%M:%S')

class TxtLogger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'a+')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        ts = get_timestamp()
        self.log_file.write(ts+ ": "+ msg + '\n')
        self.log_file.flush()
        print(msg)

    def close(self):
        self.log_file.close()