import os, re
import numpy as np
class Cio_Format(object):
    def __init__(self, Schema):
        self.vars_formats = {"TITLE": [str, 345, 1, 80],
                            "FIGFILE": [str, 7, 1, 13],
                             "NBYR": [int, 8],
                             "IYR": [int, 9],
                             "IDAF": [int, 10],
                             "IDAL": [int, 11],
                             "IGEN": [int, 13],
                             "PCPSIM": [int, 14],
                             "IDT": [int, 15],
                             "IDIST": [int, 16],
                             "REXP": [float, 17],
                             "NRGAGE": [int, 18],
                             "NRTOT": [int, 19],
                             "NRGFIL": [int, 20],
                             "TMPSIM": [int, 21],
                             "NTGAGE": [int, 22],
                             "NTTOT": [int, 23],
                             "NTGFIL": [int, 24],
                             "SLRSIM": [int, 25],
                             "NSTOT": [int, 26],
                             "RHSIM": [int, 27],
                             "NHTOT": [int, 28],
                             "WNDSIM": [int, 29],
                             "NWTOT": [int, 30],
                             "FCSTYR": [int, 31],
                             "FCSTDAY": [int, 32],
                             "FCSTCYCLES": [int, 33],
                             "RFILE(1)": [str, 35, 1, 13],
                             "RFILE(2)": [str, 35, 14, 26],
                             "RFILE(3)": [str, 35, 27, 39],
                             "RFILE(4)": [str, 35, 40, 52],
                             "RFILE(5)": [str, 35, 53, 65],
                             "RFILE(6)": [str, 35, 66, 78],
                             "RFILE(7)": [str, 36, 1, 13],
                             "RFILE(8)": [str, 36, 14, 26],
                             "RFILE(9)": [str, 36, 27, 39],
                             "RFILE(10)": [str, 36, 40, 52],
                             "RFILE(11)": [str, 36, 53, 65],
                             "RFILE(12)": [str, 36, 66, 78],
                             "RFILE(13)": [str, 37, 1, 13],
                             "RFILE(14)": [str, 37, 14, 26],
                             "RFILE(15)": [str, 37, 27, 39],
                             "RFILE(16)": [str, 37, 40, 52],
                             "RFILE(17)": [str, 37, 53, 65],
                             "RFILE(18)": [str, 37, 66, 78],
                             "TFILE(1)": [str, 39, 1, 13],
                             "TFILE(2)": [str, 39, 14, 26],
                             "TFILE(3)": [str, 39, 27, 39],
                             "TFILE(4)": [str, 39, 40, 52],
                             "TFILE(5)": [str, 39, 53, 65],
                             "TFILE(6)": [str, 39, 66, 78],
                             "TFILE(7)": [str, 40, 1, 13],
                             "TFILE(8)": [str, 40, 14, 26],
                             "TFILE(9)": [str, 40, 27, 39],
                             "TFILE(10)": [str, 40, 40, 52],
                             "TFILE(11)": [str, 40, 53, 65],
                             "TFILE(12)": [str, 40, 66, 78],
                             "TFILE(13)": [str, 41, 1, 13],
                             "TFILE(14)": [str, 41, 14, 26],
                             "TFILE(15)": [str, 41, 27, 39],
                             "TFILE(16)": [str, 41, 40, 52],
                             "TFILE(17)": [str, 41, 53, 65],
                             "TFILE(18)": [str, 41, 66, 78],
                             "SLRFILE": [str, 42, 1, 13],
                             "RHFILE": [str, 43, 1, 13],
                             "WNDFILE": [str, 44, 1, 13],
                             "FCSTFILE": [str, 45, 1, 13],
                             "PLANTDB": [str, 49, 1, 13],
                             "TILLDB": [str, 50, 1, 13],
                             "PESTDB": [str, 51, 1, 13],
                             "FERTDB": [str, 52, 1, 13],
                             "URBANDB": [str, 53, 1, 13],
                             "ISPROJ": [int, 55],
                             "ICLB": [int, 56],
                             "CALFILE": [str, 57, 1, 13],
                             "IPRINT": [int, 59],
                             "NYSKIP": [int, 60],
                             "ILOG": [int, 61],
                             "IPRP": [int, 62],
                             "IPRS": [int, 63],
                             "IPDVAR(1)": [int, 65, 1],
                             "IPDVAR(2)": [int, 65, 2],
                             "IPDVAR(3)": [int, 65, 3],
                             "IPDVAR(4)": [int, 65, 4],
                             "IPDVAR(5)": [int, 65, 5],
                             "IPDVAR(6)": [int, 65, 6],
                             "IPDVAR(7)": [int, 65, 7],
                             "IPDVAR(8)": [int, 65, 8],
                             "IPDVAR(9)": [int, 65, 9],
                             "IPDVAR(10)": [int, 65, 10],
                             "IPDVAR(11)": [int, 65, 11],
                             "IPDVAR(12)": [int, 65, 12],
                             "IPDVAR(13)": [int, 65, 13],
                             "IPDVAR(14)": [int, 65, 14],
                             "IPDVAR(15)": [int, 65, 15],
                             "IPDVAR(16)": [int, 65, 16],
                             "IPDVAR(17)": [int, 65, 17],
                             "IPDVAR(18)": [int, 65, 18],
                             "IPDVAR(19)": [int, 65, 19],
                             "IPDVAR(20)": [int, 65, 20],
                             "IPDVAB(1)": [int, 67, 1],
                             "IPDVAB(2)": [int, 67, 2],
                             "IPDVAB(3)": [int, 67, 3],
                             "IPDVAB(4)": [int, 67, 4],
                             "IPDVAB(5)": [int, 67, 5],
                             "IPDVAB(6)": [int, 67, 6],
                             "IPDVAB(7)": [int, 67, 7],
                             "IPDVAB(8)": [int, 67, 8],
                             "IPDVAB(9)": [int, 67, 9],
                             "IPDVAB(10)": [int, 67, 10],
                             "IPDVAB(11)": [int, 67, 11],
                             "IPDVAB(12)": [int, 67, 12],
                             "IPDVAB(13)": [int, 67, 13],
                             "IPDVAB(14)": [int, 67, 14],
                             "IPDVAB(15)": [int, 67, 15],
                             "IPDVAS(1)": [int, 69, 1],
                             "IPDVAS(2)": [int, 69, 2],
                             "IPDVAS(3)": [int, 69, 3],
                             "IPDVAS(4)": [int, 69, 4],
                             "IPDVAS(5)": [int, 69, 5],
                             "IPDVAS(6)": [int, 69, 6],
                             "IPDVAS(7)": [int, 69, 7],
                             "IPDVAS(8)": [int, 69, 8],
                             "IPDVAS(9)": [int, 69, 9],
                             "IPDVAS(10)": [int, 69, 10],
                             "IPDVAS(11)": [int, 69, 11],
                             "IPDVAS(12)": [int, 69, 12],
                             "IPDVAS(13)": [int, 69, 13],
                             "IPDVAS(14)": [int, 69, 14],
                             "IPDVAS(15)": [int, 69, 15],
                             "IPDVAS(16)": [int, 69, 16],
                             "IPDVAS(17)": [int, 69, 17],
                             "IPDVAS(18)": [int, 69, 18],
                             "IPDVAS(19)": [int, 69, 19],
                             "IPDVAS(20)": [int, 69, 20],
                             "IPDHRU(1)": [int, 71, 1],
                             "IPDHRU(2)": [int, 71, 2],
                             "IPDHRU(3)": [int, 71, 3],
                             "IPDHRU(4)": [int, 71, 4],
                             "IPDHRU(5)": [int, 71, 5],
                             "IPDHRU(6)": [int, 71, 6],
                             "IPDHRU(7)": [int, 71, 7],
                             "IPDHRU(8)": [int, 71, 8],
                             "IPDHRU(9)": [int, 71, 9],
                             "IPDHRU(10)": [int, 71, 10],
                             "IPDHRU(11)": [int, 71, 11],
                             "IPDHRU(12)": [int, 71, 12],
                             "IPDHRU(13)": [int, 71, 13],
                             "IPDHRU(14)": [int, 71, 14],
                             "IPDHRU(15)": [int, 71, 15],
                             "IPDHRU(16)": [int, 71, 16],
                             "IPDHRU(17)": [int, 71, 17],
                             "IPDHRU(18)": [int, 71, 18],
                             "IPDHRU(19)": [int, 71, 19],
                             "IPDHRU(20)": [int, 71, 20],
                             "ATMOFILE": [str, 73, 1, 80],
                             "IPHR": [int, 74],
                             "ISTO": [int, 75],
                             "ISOL": [int, 76],
                             "I_SUBW": [int, 77],
                             "SEPTDB": [float, 78, 1, 80],
                             "IA_B": [int, 79],
                             "IHUMUS": [int, 80],
                             "ITEMP": [int, 81],
                             "ISNOW": [int, 82],
                             "IMGT": [int, 83],
                             "IWTR": [int, 84],
                             "ICALEN": [int, 85]}

    class Schema(object):
        def __init__(self, Cio_Format):
            self.schema = {float: [], int: [], str: []}
            for i in Cio_Format.vars_formats:
                self.schema[Cio_Format.vars_formats[i][0]].append(Cio_Format.vars_formats[i][1:])


class Schema_Item(Cio_Format, Cio_Format.Schema):
    def __init__(self):
        super().__init__(self)
        self.output = []
        self.oi = None
        self.of = None

    def reader(self, k):
        for variable in self.vars_formats:
            if self.vars_formats[variable][0] == int and len(self.vars_formats[variable]) == 2:
                self.oi = int(k[self.schema[variable][0]-1].split("|")[0].replace(" ", ""))
                self.output.append([variable, self.oi])
                self.oi = None
            elif self.vars_formats[variable][0] == int and len(self.vars_formats[variable]) == 3:
                self.oi = int(k[self.schema[variable][0]-1].replace(" ", "")[self.vars_formats[variable][-1]-1])
                self.output.append([variable, self.oi])
                self.oi = None
            elif self.vars_formats[variable][0] == str:
                self.oi = str(k[self.schema[variable][0]-1][self.vars_formats[variable][1]:self.vars_formats[variable][2]].replace(" ", ""))
                self.output.append([variable, self.oi])
                self.oi = None
            elif self.vars_formats[variable][0] == float and len(self.vars_formats[variable]) == 2:
                self.oi = int(k[self.schema[variable][0]-1].split("|")[0].replace(" ", ""))
                self.output.append([variable, self.oi])
                self.oi = None
            elif self.vars_formats[variable][0] == float and len(self.vars_formats[variable]) == 4:
                self.oi = float(k[self.schema[variable][0]-1][self.vars_formats[variable][1]:self.vars_formats[variable][2]].replace(" ", ""))
                self.output.append([variable, self.oi])
                self.oi = None
