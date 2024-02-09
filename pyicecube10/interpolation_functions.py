import numpy as np


def hist_rebin(hist, new_bins):
    res = []
    amp = hist[:,2]
    old_bins = np.append(hist[:,0], max(hist[:,1]))
    bins = list(zip(np.insert(new_bins, 0, -np.inf), np.append(new_bins, np.inf)))
    iterbins = list(enumerate(old_bins))
    flag = 1
    for ledge, redge in bins[1:-1]:
        val = 0
        nledge = ledge
        for i in iterbins[flag:]:
            if ledge <= i[1] < redge:
                val += amp[i[0] - 1] * (i[1] - nledge) / (i[1] - old_bins[i[0] - 1])
                nledge = i[1]
            if redge <= old_bins[i[0] - 1]:
                break
            if redge <= i[1]:
                val += amp[i[0] - 1] * (redge - nledge) / (i[1] - old_bins[i[0] - 1])
                flag = i[0]
                break
        if (type(amp[0]) is np.ndarray) and (val is 0):
            res.append(np.zeros(len(amp[0])))
        else:
            res.append(val)
    return np.array(res)



def hist_rebin_prop(amp, old_bins, new_bins):
    res = []
    bins = list(zip(np.insert(new_bins, 0, -np.inf), np.append(new_bins, np.inf)))
    iterbins = list(enumerate(old_bins))
    flag = 1
    for ledge, redge in bins[1:-1]:
        val = 0
        prop = 0
        nledge = ledge
        for i in iterbins[flag:]:
            if ledge <= i[1] < redge:
                val += amp[i[0] - 1] * (i[1] - nledge) / (i[1] - old_bins[i[0] - 1])
                prop += np.sum(amp[i[0] - 1]) * (i[1] - nledge) / (redge - ledge)
                nledge = i[1]
            if redge <= old_bins[i[0] - 1]:
                break
            if redge <= i[1]:
                val += amp[i[0] - 1] * (redge - nledge) / (i[1] - old_bins[i[0] - 1])
                prop += np.sum(amp[i[0] - 1]) * (redge - nledge) / (redge - ledge)
                flag = i[0]
                break
        if ((type(amp[0]) is np.ndarray) and (val is 0)) or (np.sum(val) < 1e-4):
            res.append(np.zeros(len(amp[0])))
        else:
            # print(np.sum(val), prop)

            res.append(val / np.sum(val) * prop)

    # print(res)
    return np.array(res)
