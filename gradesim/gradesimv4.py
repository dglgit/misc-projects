#made by douglas lin in apcsp in like 2 hours
import numpy as np
import copy
grade_categories={'m':0.7,'p':0.3}
def calc_grade_from_list(grades):
    stacked=np.stack(grades)
    return stacked[:,0].sum()/stacked[:,1].sum()
def accumulate_grade_from_list(grades):
    stacked=np.stack(grades)
    return stacked.sum(0)
m='m'
p='p'
e='e'
def toGrade(score,total):
    return np.array([score,total])
def read_file(fname, score_delim='/',category_delim=','):
    grades={}
    with open(fname,'r') as f:
        for line in f:
            if len(line)>4:
                grade,category=line.split(category_delim)
                score,total=map(int,grade.split(score_delim))
                category=category[:-1]
                if category not in grades:
                    grades[category]=[]
                grades[category].append(toGrade(score,total))
    objects={i:GradeCategory(grades[i],grade_categories[i]) for i in grades}
    return FullGrade(MultiGrade(objects))

class GradeCategory:
    def __init__(self,grades,weighting=0.3):
        #grades should be a list of numpy size 2 arrays with format np.array([score,total])
        self.weighting=weighting
        self.grades=grades
    def accumulate(self):
        return accumulate_grade_from_list(self.grades)
    def percent_grade(self):
        return calc_grade_from_list(self.grades)
    def accumulate_(self):
        self.grades=[self.accumulate()]
    def __repr__(self):
        return f'GradeCategory<weighting: {self.weighting}, percent:{self.percent_grade()}, raw: {self.grades}>'
    def least_score_to_reach(self,percentage, total):
        n,d=self.accumulate()
        result= percentage*(total+d)-n
        return result
    def least_perfect_scores_to_reach(self, percentage):
        n,d=self.accumulate()
        result= (n-percentage*d)/(percentage-1)
        if result<0:
            print(result)
            return 'impossible'
        return result

class MultiGrade:
    def __init__(self,all_grades:dict):
        self.grades=copy.deepcopy(all_grades)
    def percent_grade(self):
        result=0
        for key in self.grades:
            category=self.grades[key]
            result+=category.weighting*category.percent_grade()
        return result
    def add_grade(self,category, score,total):
        self.grades[category].grades.append(grade(score,total))
    def __getitem__(self,key):
        return self.grades[key]
    def __repr__(self):
        start='MultiGrade: <'
        for category in self.grades:
            start+=f'{category}: {self.grades[category].grades} '
        start+='>'
        return start
    def least_score_to_reach(self,resultant, category, total):
        #when given an assignment out of `total` points, what is the least score you could get to have an overall score of `resultant`
        cum_other_scores=0
        for key in self.grades:
            if key!=category:
                cum_other_scores+=self.grades[key].percent_grade()
        least_percent=(resultant-cum_other_scores)/grade_categories[category]
        return self.grades[category].least_score_to_reach(resultant,total)
    def least_perfect_scores_to_reach(self,resultant, category):
        #when you(saveer ahem) have a bad grade and need to know how many perfect assignments you need to get your grade back to `resultant`%
        cum_other_scores=0
        for key in self.grades:
            if key!=category:
                cum_other_scores+=self.grades[key].percent_grade()
        least_percent=(resultant-cum_other_scores)/grade_categories[category]
        return self.grades[category].least_perfect_scores_to_reach(resultant)
class FullGrade:
    def __init__(self,all_grades:MultiGrade):
        #all_grades is an instance of a MultiGrade object
        
        self.branches={'root': copy.deepcopy(all_grades)}
        self.current_branch='root'
        self.grades=self.branches[self.current_branch]
        
    def percent_grade(self):
        return self.grades.percent_grade()
    def add_grade(self,category, score,total):
        self.grades.add_grade(category,score,total)
    def __getitem__(self,key):
        return self.grades[key]
    def switch_branch(self,name):
        self.branches[current_branch]=copy.deepcopy(self.grades)
        self.current_branch=name
        self.grades=self.branches[self.current_branch]
    def make_branch(self,name,switch=False):
        self.branches[name]=copy.deepcopy(self.grades)
        if switch:
            self.switch_branch(name)
    def simulate_grade(self, category, score, total):
        copied=copy.deepcopy(self.grades)
        copied.add_grade(category,score,total)
        return copied
    def __repr__(self):
        return str()
    def least_score_to_reach(self,resultant,category,total):
        return self.grades.least_score_to_reach(resultant,category,total)
    def least_perfect_scores_to_reach(self,resultant,category):
        return self.grades.least_perfect_scores_to_reach(resultant,category)
