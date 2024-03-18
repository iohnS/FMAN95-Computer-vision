def elementWiseDivision(l1, l2):
    newList = []
    if len(l1) == len(l2):
        for i in range(0, len(l1)):
            newList.append(l1[i] / l2[i])
    return newList

def pflat(data):
    newData = []
    lastCoordinates = data[len(data)-1]
    for row in data:
        newData.append(elementWiseDivision(row, lastCoordinates))
    return newData


def pflat1d(d):
    last = d[-1]
    return [e / last for e in d]