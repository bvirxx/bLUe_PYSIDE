# unused in project

""" Creates an ordered set from a list of tuples or other hashable items """

def list2OrderedSet(alist):

    mmap = {}

    oset = []

    for item in alist:
        if item not in mmap:
            mmap[item] = 1
            oset.append(item)
    return oset