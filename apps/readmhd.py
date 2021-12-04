import numpy as np
import os.path
import sys


typeidmap = {}
typeidmap['MET_SHORT'] = 'h'
typeidmap['MET_USHORT'] = 'H'
typeidmap['MET_CHAR'] = 'b'
typeidmap['MET_UCHAR'] = 'B'
typeidmap['MET_FLOAT'] = 'f'
typeidmap['MET_DOUBLE'] = 'd'

typeidmapnp2meta = {}
typeidmapnp2meta['int16'] = 'MET_SHORT'
typeidmapnp2meta['uint16'] = 'MET_USHORT'
typeidmapnp2meta['int8'] = 'MET_CHAR'
typeidmapnp2meta['uint8'] = 'MET_UCHAR'
typeidmapnp2meta['float32'] = 'MET_FLOAT'
typeidmapnp2meta['float64'] = 'MET_DOUBLE'

typeidmapmeta2np = {v:k for k,v in typeidmapnp2meta.items()} # inverted dictionary


class Volume:
    def __init__(self, vol, mhdstrs, matrixsize, voxelsize, typeid):
        self.vol = vol
        self.mhdstrs = mhdstrs
        self.matrixsize = matrixsize
        self.voxelsize = voxelsize
        self.typeid = typeid

    def __str__(self):
        return "Volume object : " +"matrixsize = " + str(self.matrixsize) + ", voxelsize = " + str(self.voxelsize) + ", typeid = " + str(self.typeid)

    def __repr__(self):
        return "Volume object\n" +"matrixsize = " + str(self.matrixsize) + "\nvoxelsize = " + str(self.voxelsize) + "\ntypeid = " + str(self.typeid) + "\nmhdstrs = " + str(self.mhdstrs) + "\nvol = " + str(self.vol)

    def writefile(self, mhdfile):
        filedir, mhdfilename = os.path.split(mhdfile)
        mhdfilenameroot, mhdfilenameext = os.path.splitext(mhdfilename)
        rawfilename = mhdfilenameroot + ".raw"
        with open(mhdfile, "wt") as f:
            f.write('ObjectType = Image\n')
            f.write('NDims = 3\n')
            f.write('Offset = 0 0 0\n')
            f.write('ElementSpacing = {0:} {1:} {2:}\n'.format(
                self.voxelsize[0], self.voxelsize[1], self.voxelsize[2]))
            f.write('DimSize = {0:} {1:} {2:}\n'.format(
                self.matrixsize[0], self.matrixsize[1], self.matrixsize[2]))
            f.write('ElementType = ' + self.typeid + '\n')
            f.write('ElementDataFile = ' + rawfilename + '\n')

        with open(os.path.join(filedir, rawfilename), "wb") as f:
            f.write(self.vol.flatten('C'))


def new(matrixsize_, voxelsize_, typeid_numpy):
    newvol = Volume(np.ndarray(matrixsize_[::-1], dtype=typeid_numpy), 
                    {}, matrixsize_, voxelsize_, typeidmapnp2meta[typeid_numpy])
    newvol.mhdstrs["ObjectType"] = "Image"
    newvol.mhdstrs["NDims"] = "3"
    newvol.mhdstrs["Offset"] = "0 0 0"
    newvol.mhdstrs["ElementSpacing"] = "{0:} {1:} {2:}".format(
        newvol.voxelsize[0], newvol.voxelsize[1], newvol.voxelsize[2])
    newvol.mhdstrs["DimSize"] = "{0:} {1:} {2:}".format(
        newvol.matrixsize[0], newvol.matrixsize[1], newvol.matrixsize[2])
    newvol.mhdstrs["ElementType"] = newvol.typeid
    return newvol


def read(mhdfilename, readrawfile=True):

    mhdstrs = {}
    for line in open(mhdfilename, 'rt'):
        linelist = line.split("=")
        tagid = linelist[0].strip()
        tagcontent = linelist[1].strip()
        mhdstrs[tagid] = tagcontent
        if(tagid.lower() == "dimsize"):
            matrixsize = list(map(int, tagcontent.split(' ')))
        if(tagid.lower() == "elementspacing"):
            voxelsize = list(map(float, tagcontent.split(' ')))
        if(tagid.lower() == "elementtype"):
            typeid = tagcontent
    
    (mhdfileroot, mhdfileext) = os.path.splitext(mhdfilename)
    rawfilename = mhdfileroot + '.raw'
    
    if readrawfile:
        vol = np.fromfile(rawfilename, typeidmapmeta2np[typeid])
        vol = vol.reshape(matrixsize[::-1])     # ********** NOTE: vol[z][y][x]
    else:
        vol = []
    
    res = {}
    res['vol'] = vol
    res['mhdstrs'] = mhdstrs
    res['matrixsize'] = matrixsize
    res['voxelsize'] = voxelsize
    res['typeid'] = typeid
    
    return Volume(vol, mhdstrs, matrixsize, voxelsize, typeid)



    