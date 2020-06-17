import os, warnings , re

class Read_Files:
    def __init__(self, check=True, Direction=input("Dir : ")):
        self.Direction = Direction
        self.files = os.listdir(self.Direction)
        self.check = check

    def gather_file_names(self):
        def readfile(file):
            self.content = open(file).readlines()
            return self.content

        if self.check:
            if "fig.fig" in self.files:
                self.fig_content = readfile("fig.fig")
                self.num_subs = 0
                for line in self.fig_content:
                    if line[15]=='1':
                        self.num_subs +=1
                self.name_subs = [self.fig_content[i].replace(' ', '').split()[0] for i in range(1, 2*self.num_subs, 2)]
            else:
                warnings.warn('fig.fig file is not available!')
        self.hru_level_file_names = [[re.findall('.{1,13}', open(j).readlines()[i]) for i in range(61, 65)] for j in self.name_subs]


    def check_files(self):
        if self.check:
            for i in self.hru_level_file_names:
                if i not in self.files:
                    warnings.warn(i + 'is not available in your directory')