# This is an improved version of
# https://stackoverflow.com/questions/32163436/python-decorator-for-printing-every-line-executed-by-a-function
# Modifiactions (BV)
# - added a profiler
# - corrected a bug in __exit__
# usage : add decorator @tdec to the function to profile

import sys
from os.path import basename
from time import time

class res(object):
    def __init__(self, line_no, etime):
        self.line_no, self.etime = line_no, etime
    def toStr(self):
        return '%d: %.5f' % (self.line_no, self.etime)

class debug_context():
    """ Debug context to trace any function calls inside the context """

    def __init__(self, name):
        self.line_no, self.prev_line_no  = -1, -1
        self.name = name
        self.t = 0.0
        self.done = False # used in trace_calls to prevent nested traces

    def __enter__(self):
        print('Debug/Profile %s ' % self.name, end='')
        # Set the trace function to the trace_calls function
        # So all events are now traced
        sys.settrace(self.trace_calls)
        self.t = time()
        self.line_no = -1
        self.out = []
        self.ttime = 0.0

    def __exit__(self, *args, **kwargs):
        # Stop tracing all events
        sys.settrace(None)
        # get cumulated time for each line_no
        d = dict()
        for item in self.out:
            if item.line_no in d:
                d[item.line_no].etime += item.etime
            else:
                d[item.line_no] = item
        result = [item.toStr() for item in sorted(list(d.values()), key=lambda x: x.etime, reverse=True)]
        print('\n'.join(result))
        print('************ cumul : %.5f s' % self.ttime)

    def trace_calls(self, frame, event, arg):
        # no nested traces
        if self.done:
            return
        # We want to only trace our call to the decorated function
        elif event != 'call':
            return
        elif frame.f_code.co_name != self.name:
            return
        self.done = True
        print('in file %s :' % basename(frame.f_code.co_filename))
        print('cumulated line  time (s) (sorted by decreasing values)')
        # return the trace function to use when you
        # go into that function call
        return self.trace_lines

    def trace_lines(self, frame, event, arg):
        # If you want to print local variables each line
        # keep the check for the event 'line'
        # If you want to print local variables only on return
        # check only for the 'return' event
        # called before executing line
        if event not in ['line', 'return']:
            return
        tsave = self.t
        self.t = time()
        co = frame.f_code
        func_name = co.co_name
        filename = co.co_filename
        local_vars = frame.f_locals
        # save previous line number and get current line number
        self.prev_line_no = self.line_no
        self.line_no = frame.f_lineno
        if self.prev_line_no > -1:
            et = self.t - tsave
            self.out += [res(self.prev_line_no, et)]
            self.ttime += et

def tdec(func):
    """
    Debug/Profile decorator
    """
    def decorated_func(*args, **kwargs):
        with debug_context(func.__name__):
            return_value = func(*args, **kwargs)
        return return_value
    return decorated_func