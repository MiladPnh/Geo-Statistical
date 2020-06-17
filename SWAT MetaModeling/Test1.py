__author__ = 'MP'
from xml.etree.ElementTree import Element, Comment, SubElement, tostring
from xml.dom import minidom
import xml.etree.ElementTree as Et
import re
class FileCio_Format(object):

    def vars(self):
        #Readfile.readfile()
        with open('file.cio') as o:
            #by re gotta be splited by len = 13
            #to approach the RFILE elements one should assign the specific num of intended Rfile
            self.RFILE = ['RFILE']
            self.RFILE.append(re.findall('.{1,13}', o.readlines()[34]))
            self.RFILE.append(re.findall('.{1,13}', o.readlines()[35]))
            self.RFILE.append(re.findall('.{1,13}', o.readlines()[36]))
            #to approach the TFILE elements one should assign the specific num of intended Tfile
            self.TFILE = ['TFILE']
            self.TFILE.append(re.findall('.{1,13}', o.readlines()[38]))
            self.TFILE.append(re.findall('.{1,13}', o.readlines()[39]))
            self.TFILE.append(re.findall('.{1,13}', o.readlines()[40]))
            # self.li = ['comment', 'variable name']

            self.l = ['',
            'Master Watershed File: file.cio',
            'Project Description:',
            'General Input/Output section (file.cio):',
            '10/25/2011 12:00:00 AM ARCGIS-SWAT interface AV',
            '',
            'General Information/Watershed Configuration:',
            ['FIGFILE', re.findall('.{1,13}', o.readlines()[6])],
            ['Number of years simulated', 'NBYR'],
            ['Beginning year of simulation', 'IYR'],
            ['Beginning julian day of simulation', 'IDAF'],
            ['Ending julian day of simulation', 'IDAL'],
            'Climate:',
            ['Random number seed cycle code', 'IGEN'],
            ['precipitation simulation code: 1=measured, 2=simulated', 'PCPSIM'],
            ['Rainfall data time step', 'IDT'],
            ['rainfall distribution code: 0 skewed, 1 exponential', 'IDIST'],
            ['Exponent for IDIST=1', 'REXP'],
            ['number of pcp files used in simulation', 'NRGAGE'],
            ['number of precip gage records used in simulation', 'NRTOT'],
            ['number of gage records in each pcp file', 'NRGFIL'],
            ['temperature simulation code: 1=measured, 2=simulated', 'TMPSIM'],
            ['number of tmp files used in simulation', 'NTGAGE'],
            ['number of temp gage records used in simulation', 'NTTOT'],
            ['number of gage records in each tmp file', 'NTGFIL'],
            ['Solar radiation simulation Code: 1=measured, 2=simulated', 'SLRSIM'],
            ['number of solar radiation records in slr file', 'NSTOT'],
            ['relative humidity simulation code: 1=measured, 2=simulated', 'RHSIM'],
            ['number of relative humidity records in hmd file', 'NHTOT'],
            ['Windspeed simulation code: 1=measured, 2=simulated', 'WINDSIM'],
            ['number of wind speed records in wnd file', 'NWTOT'],
            ['beginning year of forecast period', 'FCSTYR'],
            ['beginning julian date of forecast period', 'FCSTDAY'],
            ['number of time to simulate forecast period', 'FCSTCYCLES'],
            'Precipitation Files:',
            self.RFILE,
            'Temperature Files:',
            self.TFILE,
            ['name of solar radiation file', 'SLRFILE'],
            ['name of relative humidity file', 'RHFILE'],
            ['name of wind speed file', 'WNDFILE'],
            ['name of forecast data file', 'FCSTFILE'],
            'Watershed Modeling Options:',
            ['name of basin input file', 'BSNFILE'],
            'Database Files:',
            ['name of plant growth database file', 'PLANTDB'],
            ['name of tillage database file', 'TILLDB'],
            ['name of pesticide database file', 'PESTDB'],
            ['name of fertilizer database file', 'FERTDB'],
            ['name of fertilizer database file', 'FERTDB'],
            'Special Projects:',
            ['special project: 1=repeat simulation', 'ISPROJ'],
            ['auto-calibration option: 0=no, 1=yes', 'ICLB'],
            ['auto-calibration parameter file', 'CALFILE'],
            'Output Information:',
            ['print code (month, day, year)', 'IPRINT'],
            ['number of years to skip output printing/summarization', 'NYSKIP'],
            ['streamflow print code: 1=print log of streamflow', 'ILOG'],
            ['print code for output.pst file: 1= print pesticide output', 'IPRP'],
            ['print code for final soil chemical data (.chm format)', 'IPRS'],
            'Reach output variables:',
            #self.l[][1][i] & i for the num of intended var
            ['IPDVAR'],
            'Subbasin output variables:',
            ['IPDVAB'],
            'HRU output variables:',
            ['IPDVAS'],
            'HRU data to be printed:',
            ['IPDHRU'],
            #variables from source file
            #i for variable name & j for specific file num for exmple self.l65[0][1] return the IPDVAR/B/S(1)
            'ATMOSPERIC DEPOSITION',
            '',
            ['print code for hourly output 0=no 1=yes (hourq.out)', 'IPHR'],
            ['print code for soil storage 0=no 1=yes (soilst.out)', 'ISTO'],
            ['Code for printing phosphorus/nitrogen in soil profile (output.sol)', 'ISOL'],
            ['Code for routing headwaters', 'I_SUBW'],
            '',
            ['Code for binary output of files (.rch, .sub, .hru files only)', 'IA_B'],
            ['Print watqual.out file 0=no 1=yes', 'IHUMUS'],
            ['0=print no file(s) 1=print tempvel and tempdep', 'ITEMP'],
            ['0=do not print snowband.out; 1=print snowband.out', 'ISNOW'],
            ['0=do not print output.mtg; 1=print output.mgt', 'IMGT'],
            ['0=do not print files; 1=print files', 'IWTR'],
            ['0=print Julian day; 1=print calender day mm/dd/yyyy', 'ICALEN']]




    class schema(object):
        comments = [1, 2, 3, 4, 6, 12, 34, 38, 46, 48, 54, 58, 64, 66, 68, 70, 72]
        simplecharvarlines = [7, 42, 43, 44, 45, 47, 49, 50, 51, 52, 53, 57]
        simpledigitvarlines = [8, 9, 10, 11, 13, 14, 15, 16, 17, 18,
                               19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                               29, 30, 31, 32, 33, 55, 56, 59, 60, 61,
                               62, 63, 74, 75, 76, 77, 79, 80, 81, 82, 83]
        multiplecharvarlines = [35, 39]
        multipledigitvarlines = [65, 67, 69, 71]
        #it should be contained of schema of line patterns??? ==> by determinning the type of lines which is based on simple or multiple vars
###for each lines gotta use different function of schema_item class

        #append all read datas from SWAT files to lists which aforedefined
        #here it is where we must define the format of our variable


    #class OuterClass:
    #outer_var = 1
    #class InnerClass:
        #def __init__(self):
            #self.inner_var = OuterClass.outer_var
class schema_item(FileCio_Format):
    def simpledigitvarlines(self, i):
        with open('file.cio') as o:
            if i == 17:
                self.contentsd = int(o.readlines()[i - 1].split("|")[0].replace(" ", ""))
            else:
                self.contentsd = float(o.readlines()[i - 1].split("|")[0].replace(" ", ""))

    def simplecharvarlines(self, i):
        with open('file.cio') as o:
            self.contentsch = re.findall('.{1,13}', o.readlines()[i - 1])[0]

    def multiplecharvarlines(self, i): #RFILE & TFILE
        with open ('file.cio') as o:
            self.contentmch = []
            self.contentmch.append(re.findall('.{1,13}', o.readlines()[i - 1]))
            self.contentmch.append(re.findall('.{1,13}', o.readlines()[i]))
            self.contentmch.append(re.findall('.{1,13}', o.readlines()[i + 1]))


    def multipledigitvarlines(self, i):  #lines:65,67,69,71
        with open('file.cio') as o:
            self.contentmd = o.readlines()[i - 1].replace(" ", "_").split("___")
            #self.contentmd[1] = IPDVAR #1 for example


    def specificdataline(self, i, j):
        self.contentsd = []
        with open('000010001.mgt.xml') as o:
            s = o.readlines()[i-1]
            self.o = []
            while s:
                k = j+3
                self.o.append(s[:k])
                s = s[k:]
        self.contentsd.append(self.o[])


        self.contentsd.append(self.o[i])




class Readfile(schema_item):
    def getfilename(self):
        import os
        self.path = input("Please import the direction of SWAT files")
        self.dirs = os.listdir(self.path)

    def __init__(self):
        print("Welcome to the SWAT")
        self.getfilename()
        self.checkfilename()

    def checkfilename(self):
        import re
        with open('fig.fig') as o:
            self.list = []
            n = 0
            for i in o.readlines():
                if i[2] == 'b':
                    n +=1
            #print(n) n = num of subbasins

            for i in range(1, 2*n, 2):
                self.list.append(o.readlines()[i].replace(' ', ''))


        for i in range(0, n):
            with open(self.list[i]) as l:
                self.content = l.readlines()

                for f in range(61, 65):
                    self.n = re.findall('.{1,13}', self.content[f])
                    self.n = self.n.append('file.cio')
                #check coincidence
                    self.nm = set(self.n)
                    for k in self.n:

                        if k in self.dirs:
                            print('ok')
                            break
                        else:
                            print('the number of this message plus 1 = the number of specific HRU')
                    break
            self.readfile()


    def readfile(self):
        c.__init__()
        with open(self.name) as o:
            self.contentt = o.readlines()


#class Xmldesign(Readfile):
    def writeintoxml1(self):
        r = self.xml_design1()
        t = open('Cio.xml', 'w').write(r)



    def writeintoxml(self):
        r = self.xml_design()
        t = open('inputfile1.xml', 'w').write(r)

    def xml_design(self):
        self.top = Element('file.cio')
        #x = [self.p, self.etmax, self.LU, self.HG, self.A]
        #y = ['Precipitation', 'maximum_Evaporation', 'Land_use', 'Hydraulic_group',
             #'Area_square_meter_']
        #z = ['P', 'etmax', 'LU', 'HG', 'A']

        for i in range(0, len(self.)):
            value = x[i]
            key = y[i]
            sub = z[i]
            comment = Comment(key)
            self.top.append(comment)
            child = SubElement(self.top, sub)
            child.text = str(value)

        return self.pretty(tostring(self.top))

    def pretty(self, xmls):
        xml = minidom.parseString(xmls)
        return xml.toprettyxml()


    def readxmlfile(self):
        tree = Et.parse('inputfile.xml')
        root = tree.getroot()
        for child in root:
            self.tag = child.tag
            assert isinstance(child.text, object)
            self.data = child.text



















