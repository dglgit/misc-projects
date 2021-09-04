import numpy as np
import copy
# a grade file should be a csv type file with format {score}/{total},{type}
# where type is m for mastery, p for progression, and e for engagement
weights={'p':.3,'e':0,'m':.7}
global_prefix='logs/'
def collapse_grade(x):
    return 100*x[0]/x[1]
def fmt_breakdown(d):
    meaned=d.mean(0)
    print(d,meaned)
    avg=collapse_grade(meaned)
    return f'{str(d)}:{avg}'
def array_append(src,new):
    return np.append(src,new).reshape(-1,2)
class Gradefile:
    def __init__(self,fname,bypass=None):
        if bypass is None:
            self.fname=fname
            self.grades={'p':[],'e':[],'m':[]}
            with open(global_prefix+fname,'r') as gf:
                for line in gf:
                    grade,category=line.split(',')
                    score,total=grade.split('/')
                    grade=np.array([float(score),float(total)])
                    self.grades[category[0]].append(grade)
            for c in self.grades:
                if len(self.grades[c])>0:
                    self.grades[c]=np.stack(self.grades[c])#2d array
                else:
                    self.grades[c]=np.array([[0,0]])
            self.branches={'origin':self.grades}
            self.branch_name='origin'
        else:
            self.grades=bypass
    def __getitem__(self,x):
        return self.grades[x]
    def __repr__(self):
        new_stuff={c:self.grades[c].tolist() for c in self.grades}
        return f'<{str(new_stuff)},{self.fname},branch:{self.branch_name}>'

    #def __str__(self):
    #    return f'<{str(self.grades)},{self.fname},branch:{self.branch_name}>'
    def calculate(self):
        total=0
        for c in self.grades:
            cg=collapse_grade(self.grades[c].sum(0))*weights[c]
            if not np.isnan(cg):
                total+=cg
        return total
    def breakdown(self):
        means={c:fmt_breakdown(self.grades[c]) for c in self.grades}
        overall=self.calculate()
        return f'{means},overall:{overall}'
    def fake_add(self,score,total,category):
        copied=copy.copy(self.grades)
        fake_grade=np.array([score,total])
        copied[category]=np.append(copied[category],fake_grade).reshape(-1,2)
        surrogate=Gradefile('',bypass=copied)
        return surrogate.breakdown()
    def make_branch(self,name):
        assert name !='origin'
        copied=copy.copy(self.grades)
        self.branches[name]=copied
        self.branch_name=name
        self.grades=copied
    def switch_branch(self,name):
        self.branches[self.branch_name]=self.grades
        self.branch_name=name
        self.grades=self.branches[name]
    def add_grade(self,score,total,category):
        self.grades[category]=array_append(self.grades[category],np.array([score,total]))
    def delete_branch(self,name):
        if self.branch_name==name:
            self.switch_branch('origin')
        del self.branches[name]
    def update_file(self):
        with open(self.fname,'w') as f:
            for c in self.grades:
                g=self.grades[c]
                for row in g:
                    score,total=row
                    line=f'{score}/{total},{c}\n'
                    f.write(line)

m='m'
p='p'
e='e'
doc='''
'''
if __name__=='__main__':
    test=Gradefile('logs/new_test')
    print(test.breakdown())
    test.add_grade(23,24,'m')
    print(test.breakdown())
