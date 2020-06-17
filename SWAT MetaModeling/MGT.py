__author__ = 'My PC'
from xml.etree.ElementTree import Element, Comment , SubElement ,tostring
from xml.dom import minidom

class mgtObject:

    def __init__(self):
        self.top = Element('mgt')

    def pretty(self, xmls):
        xml = minidom.parseString(xmls)
        return xml.toprettyxml()

    def __designXml__(self, param, num, comment):
        if comment is not " ":
            comment = Comment(comment)
            self.top.append(comment)
        child = SubElement(self.top, param.replace(" ",""))
        child.text = str(num)

    def print_xml(self):
        print(self.pretty(tostring(self.top)))


    def read_file(self):

        with open('000010001.mgt.xml') as f:
            content = f.readlines()
        attr = content[0]
        self.top.set("detail", attr)
        for line in content[1:]:
            x = line.split("|")

            if x[0] == line:
                header = line.replace(" ","_")
                # if line=="Operation_Schedule:":
                #    break
        #         print(header)
                continue

            num = x[0]
            param = x[1].split(":")[0]
            comment = x[1].split(":")[1]

            num = float(num)
            self.__designXml__(param, num, comment)

        self.print_xml()

mgt_object = mgtObject()
mgt_object.read_file()





