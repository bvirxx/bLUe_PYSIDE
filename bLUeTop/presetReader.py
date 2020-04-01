"""
This File is part of bLUe software.

Copyright (C) 2017  Bernard Virot <bernard.virot@libertysurf.fr>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Lesser Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import struct

import cv2
import numpy as np
from PySide2.QtCore import QRect, QPoint
from PySide2.QtWidgets import QApplication


class aTaggedBlock():
    """
    Container for Adobe Photoshop tagged block. Tagged blocks are described in the
    section "Additional Layer Information" of the specification :
    see https://www.adobe.com/devnet-apps/photoshop/fileformatashtml
    A tagged block is a sequence of sub-blocks :
    each sub-block begins with 4 bytes indicating the length of its data,
    followed by a header and a "Virtual Memory Array List" (VMAL).
    The structure of the header depends on the block tag (samp, patt,...)

    """

    headerSize = 12  # signature (4), tag(4), size of data (4)

    def __init__(self, addr, lg, signature, tag):
        if not signature in ['8BIM', '8B64']:
            raise ValueError('bad block signature')
        if not tag in ['samp', 'patt', 'pat2', 'pat3', 'desc']:
            raise ValueError('bad block tag')
        self.addr = addr
        self.length = lg
        self.signature = signature
        self.tag = tag

class aParser():
    """
    Parser for Adobe brush preset files (.abr).
    Currently, only file version 6.2 is supported.
    See http://fileformats.archiveteam.org/wiki/Photoshop_brush
    A partial specification of the format can be found in
    https://www.adobe.com/devnet-apps/photoshop/fileformatashtml
    After the version number (4 bytes), the file is a sequence
    of tagged blocks.
    Each tagged block is a sequence of sub-blocks :
    each sub-block begins with 4 bytes indicating the length of its data,
    followed by a header and a "Virtual Memory Array List" (VMAL).
    The structure of the header depends on the block tag (samp, patt,...)
    """

    @staticmethod
    def getVersion(filename):
        try:
            with open(filename, "rb") as f:
                buf = f.read()
            v = struct.unpack(">2h", buf[:4])
        except IOError as e:
            print("cannot open %s" % filename, e)
            raise
        except struct.error:
            print("Cannot find file version")
            raise
        return "%d.%d" % v

    @staticmethod
    def align4(addr):
        r = addr % 4
        if r == 0:
            return addr
        else:
            return (addr // 4) * 4 + 4

    @staticmethod
    def readPString(buf):
        """
        gets  Pascal string from buf
        @param buf:
        @type buf:
        @return:
        @rtype:
        """
        lg = struct.unpack(">B", buf[:1])[0]
        id = struct.unpack(">%ds" % lg, buf[1:1 + lg])
        bytesRead = lg + 1
        return id, bytesRead

    @staticmethod
    def readUString(buf):
        """
        Grts unicode string from buf
        @param buf:
        @type buf:
        @return: string, number of bytes read
        @rtype: 2-uple
        """
        lg = struct.unpack(">l", buf[:4])[0]
        try:
            id = buf[4:4 + 2 * (lg - 1)].decode(encoding="utf-8")  # utf-16 yields strange things
        except UnicodeDecodeError as e:
            print(e)
        return id, 2 * lg + 4

    @staticmethod
    def findTaggedBlocks(buf):
        """
        Returns the list of addresses of the tagged blocks contained in buf
        @param buf: .abr data
        @type buf: bytes
        @return: list of tagged blocks
        @rtype: list of int
        """
        # skip file version major (2) and minor (2)
        next = 4
        tBlocks = []
        try:
            while next < len(buf):  #True:
                start = next
                try:
                    # get block signature : should be 8BIM or 8B64
                    signature = buf[next : next+4].decode("utf-8")
                    next += 4
                    tag = buf[next : next+4].decode("utf-8")
                    next += 4
                except UnicodeDecodeError:
                    pass
                # get block length
                lg = struct.unpack(">l", buf[next : next + 4])[0]
                next += lg + 4
                tBlocks.append(aTaggedBlock(start, lg, signature, tag))
        except struct.error as e:
            print(e)
        return tBlocks

    @staticmethod
    def findSubBlocks(buf, tBlock):
        """
        Returns the addresses of  the sub-blocks
        contained in tBlock
        Each sub-block begins with the length (4 bytes) of its data.
        @param buf: .abr data buffer
        @type buf: bytes
        @param tBlock: tagged block
        @type tBlock: aTaggedBlock
        @return: list of VMAL addresses
        @rtype: list of int
        """
        next = tBlock.addr + aTaggedBlock.headerSize
        blocks = [next]
        try:
            while True: # next < tBlock.addr + tBlock.length:
                # get length of VMAL data
                lg = struct.unpack('>l', buf[next:next+4])[0]
                next += lg + 4
                next = aParser.align4(next)
                if next < tBlock.addr + tBlock.length + 12:  # +12 added 31/03/20 (header)
                    blocks.append(next)
                else:
                    break
        except struct.error:
            # no more data : exit loop
            pass
        return blocks

    @staticmethod
    def readSubSamp(buf):
        """
        Parses a "samp" sub-block and returns the correspondiing VMAL
        @param buf: data of "samp" sub-block (4-bytes header excluded)
        @type buf: bytes
        @return: VMAL
        @rtype: preset
        """
        prst = preset()
        prst.id, s = aParser.readPString(buf)
        print('samp sub-block id : ', prst.id)
        # skip 4 bytes 00 01 00 00
        s1 = prst.readVMALHeader(buf[s + 4:])
        print(prst.VMALHeader)
        offset = s + 4 + s1
        print('channelCount', prst.channelCount)
        # init preset vmaList
        for i in range(prst.channelCount):
            count = prst.readVMA(buf[offset:])
            offset += count
        # build image buffer for each vma in preset
        for i, vma in enumerate(prst.vmaList):
            w, h = vma.rectangle.width() - 1, vma.rectangle.height() - 1
            if vma.compressionMode == 1:
                out = aParser.decompressBitmap(vma.data, w, h)
            else:
                out = np.frombuffer(vma.data[ : w * h], dtype = np.uint8)
            out = out.reshape((vma.rectangle.height() - 1, vma.rectangle.width() - 1))
            vma.imgBuf = out
        return prst

    @staticmethod
    def readSubPatt(buf):
        """
        Reads a sub-block of a "patt" tagged block and returns the
        corresponding VMAL
        @param buf:
        @type buf:
        @return: VMAL
        @rtype: preset
        """
        # skip header
        id, s = aParser.readUString(buf[12:])
        id1, s1 = aParser.readPString(buf[12 + s:])
        # next is one VMAL
        prst= preset()
        s2 = prst.readVMALHeader(buf[12+s+s1:])
        print(prst.VMALHeader)
        offset = 12 + s + s1 +s2
        print('channelCount', prst.channelCount)
        # init preset VMAList
        for i in range(prst.channelCount):
            count = prst.readVMA(buf[offset:])
            offset += count
            vma = prst.vmaList[-1]
        # build image buffer for each vma in preset
        for i, vma in enumerate(prst.vmaList):
            w, h = vma.rectangle.width() - 1, vma.rectangle.height() - 1
            if vma.compressionMode == 1:
                out = aParser.decompressBitmap(vma.data, w, h)
            else:
                out = np.frombuffer(vma.data[: w * h], dtype=np.uint8)
            out = out.reshape((vma.rectangle.height() - 1, vma.rectangle.width() - 1))
            vma.imgBuf = out
        return prst

    @staticmethod
    def decompressBitmap(buf, w, h):
        out = np.empty((w * h,), dtype=np.uint8)
        out_next = 0  # offset of next value to write
        offset = 0  # offset of next byte to read
        # read compressed line widths (short unsigned ints)
        compWidths = struct.unpack('>%dh' % h, buf[: 2 * h])
        offset = 2 * h
        # read lines
        for i in range(h):
            #if len(buf[offset:]) <compWidths[i]: # == 0:
                #break
            j = 0
            while j < compWidths[i]:
                n = struct.unpack('>B', buf[offset:offset + 1])[0]
                j += 1
                offset += 1
                if n >= 128:
                    n -= 256
                if n < 0:
                    if n == -128:
                        continue
                    n = -n + 1
                    c = struct.unpack('>B', buf[offset:offset + 1])[0]
                    j += 1
                    offset += 1
                    out[out_next:out_next + n] = c
                    out_next += n
                else:  # copy n+1 bytes
                    count = n + 1
                    out[out_next:out_next + count] = np.frombuffer(buf[offset:offset + count], dtype=np.uint8)
                    #out[out_next:out_next + count] = struct.unpack('>%dB' % count, buf[offset:offset + count])
                    out_next += count
                    offset += count
                    j += count
        return out

    @staticmethod
    def readFile(filename):
        with open(filename, "rb") as f:
            buf = f.read()
        print(' raed bytes ', len(buf))

        taggedBlocks = aParser.findTaggedBlocks(buf)
        images = []
        for tb in taggedBlocks:
            if tb.tag == 'desc':
                continue
            blocks = aParser.findSubBlocks(buf, tb)
            try:
                if tb.tag == 'samp':
                    for addr in blocks:
                        prst = aParser.readSubSamp(buf[addr + 4:])  # skip header : sub-block size (4 bytes)
                        images.extend([vma.imgBuf for vma in prst.vmaList])
                elif tb.tag == 'patt':
                    for addr in blocks:
                        prst = aParser.readSubPatt(buf[addr + 4:])  # skip header : sub-block size (4 bytes)
                        images.extend([vma.imgBuf for vma in prst.vmaList])
            except ValueError as e:
                print(e, addr)
        return images


class aVMA:
    """
    Adobe virtual memory array
    See https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/
    """
    def __init__(self, length=0, pixelDepth=0, rectangle=None, compressionMode=0, data=None):
        self.length = length
        self.pixelDepth = pixelDepth
        self.rectangle = rectangle
        self.compressionMode = compressionMode
        self.data = data


class preset:
    """
    A preset corresponds to a VMAL. It is a wrapper
    for the the list of vma objects extracted from the VMAL.
    """

    def __init__(self, name=''):
        self.vmaList = []
        self.name = name

    def readVMALHeader(self, buf):
        format = ">7l"
        self.VMALHeader = struct.unpack(format, buf[:28])
        self.channelCount = self.VMALHeader[6]
        return 7 * 4

    def readVMA(self, buf):
        """
        Parses a VMA data buffer into a newly created
        vma object and add the object to self.vmaList.
        Returns the (total) size of VMA data
        @param buf: VMA data
        @type buf: bytes
        @return: VMA total size
        @rtype: int
        """
        # get header
        format = ">lll4lhB"
        headerSize = 31
        VMAHeader = struct.unpack(format, buf[:headerSize])
        if VMAHeader[0] == 0:  # skip
            return 4
        # get length of following VMA data: total size is lg + 8
        lg = VMAHeader[1]
        if lg == 0:
            return 8  # skip
        pixelDepth = VMAHeader[2]
        if pixelDepth != 8:
            raise ValueError('bad pixel depth')
        rectangle = QRect(QPoint(VMAHeader[4], VMAHeader[3]), QPoint(VMAHeader[6], VMAHeader[5]))  # 3=top, 4=left, 5=bottom, 6=right
        pixelDepth1 = VMAHeader[7]
        compressionMode = VMAHeader[8]
        vma = aVMA(length=lg, pixelDepth=pixelDepth, rectangle=rectangle, compressionMode=compressionMode, data=buf[headerSize: headerSize + lg])
        self.vmaList.append(vma)
        return lg + 8

if __name__ == "__main__":
   # path = "C:\\users\\berna\\desktop\\Retouching+Brushes.abr"
    path = "C:\\users\\berna\\desktop\\20 WaterFall Brushes.abr"
    version = aParser.getVersion(path)
    print("version :" , version)
    images = aParser.readFile(path)
    print ("%d images found" % len(images))
    for i, im in enumerate(images):
        print('image size :', im.shape[1], im.shape[0])
        cv2.namedWindow('toto%d' %i, cv2.WINDOW_NORMAL)
        cv2.resizeWindow('toto%d' %i, im.shape[1], im.shape[0])
        cv2.imshow('toto%d' %i, im)
        cv2.waitKey(1000)
