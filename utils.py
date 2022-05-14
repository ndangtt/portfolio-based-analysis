import pandas as pd
import numpy as np
import sys
import json
import shutil
import subprocess
import os
import glob
from functools import cmp_to_key
import itertools
import ast

def calculate_minizinc_borda_scores(status1:str, status2:str, time1:float, time2:float, problemType:str, objs1:list=[], objs2:list=[], zeroScoreWhenBothFail:bool=False):
    """
    Compute MiniZinc competition's Borda scores between two runs of two solvers.
    There are two scoring methods in this category: complete/incomplete. See Assessment section in https://www.minizinc.org/challenge2021/rules2021.html for more details.
    Important note: When both solvers fail to solve an instance, the competition scoring procedure will give a score of 0 to solver1 and a score of 1 to solver2. To give both solvers a score of 0 in such case, set zeroScoreWhenBothFail=True

    """
    assert status1 in ['S','C','ERR','UNK','MZN','INC'], status2 in ['S','C','ERR','UNK','MZN','INC']
    assert problemType in ['MIN','MAX','SAT']

    scores = {'complete':(), 'incomplete':()}

    def solved(status):
        return status in ['S','C']
    
    def has_better_objective(o1, o2, problemType):
        assert problemType in ['MIN','MAX']
        if problemType=='MIN':
            return o1<o2
        return o1>o2

    # for decision problems, the two scoring methods are the same
    if problemType=='SAT':

        def better_sat(s1,s2):
            if solved(s1) and not solved(s2):
                return True
            return False

        # instance is solved by only one of the two solvers
        if better_sat(status1, status2):
            scores['complete'] = scores['incomplete'] = (1,0)
        elif better_sat(status2, status1):
            scores['complete'] = scores['incomplete'] = (0,1)
        # instance is solved/unsolvable by both solvers
        else:
            # instance is solved by both solvers
            if solved(status1) and solved(status2):
                # TODO: the competition rules say "0.5 if both finished in 0s", but how to check it?
                scores['complete'] = scores['incomplete'] = (time2/(time1+time2), time1/(time1+time2))
            # instance is unsolvable for both solvers
            else:
                assert (not solved(status1)) and (not solved(status2))
                if zeroScoreWhenBothFail:
                    scores['complete'] = scores['incomplete'] = (0,0)
                else:
                    scores['complete'] = scores['incomplete'] = (0,1)

    # calculate scores for optimisation problems
    else:

        # complete scoring
        def better_optimisation_complete(s1,s2,o1,o2):
            if solved(s1) and not solved(s2):
                return True
            if (s1=='C') and (s2!='C'):
                return True
            if (s1==s2) and (s1=='S'):
                assert len(o1)>0 and len(o2)>0
                lastObj1 = o1[-1][1]
                lastObj2 = o2[-1][1]
                return has_better_objective(lastObj1, lastObj2, problemType)
            return False
        if better_optimisation_complete(status1, status2, objs1, objs2):
            scores['complete'] = (1,0)
        elif better_optimisation_complete(status2, status1, objs2, objs1):
            scores['complete'] = (0,1)
        else:
            # both solvers fail
            if (not solved(status1)) and (not solved(status2)):
                if zeroScoreWhenBothFail:
                    scores['complete'] = (0,0)
                else:
                    scores['complete'] = (0,1)
            # both solvers complete
            elif (status1=='C') and (status2=='C'):
                scores['complete'] = (time2/(time1+time2), time1/(time1+time2))
            # both solvers give equal solution quality but without optimality proof
            else:
                assert (status1=='S') and (status2=='S')
                assert objs1[-1][1]==objs2[-1][1]
                lastTime1 = objs1[-1][0]
                lastTime2 = objs2[-1][0]
                scores['complete'] = (lastTime2/(lastTime1+lastTime2), lastTime1/(lastTime1+lastTime2))

        # incomplete scoring
        def better_optimisation_incomplete(s1,s2,o1,o2):
            if solved(s1) and not solved(s2):
                return True
            if solved(s1) and solved(s2) and len(o1)>0:
                assert len(o2)>0
                lastObj1 = o1[-1][1]
                lastObj2 = o2[-1][1]
                return has_better_objective(lastObj1, lastObj2, problemType)
            return False
        if better_optimisation_incomplete(status1, status2, objs1, objs2):
            scores['incomplete'] = (1,0)
        elif better_optimisation_incomplete(status2, status1, objs2, objs1):
            scores['incomplete'] = (0,1)
        else:
            # both solvers fail
            if (not solved(status1)) and (not solved(status2)):
                if zeroScoreWhenBothFail:
                    scores['incomplete'] = (0,0)
                else:
                    scores['incomplete'] = (0,1)
            # both solvers complete
            elif (status1=='C') and (status2=='C'):
                scores['incomplete'] = (time2/(time1+time2), time1/(time1+time2))
            # both solvers give equal solution quality
            else:
                assert solved(status1) and solved(status2)
                assert objs1[-1][1]==objs2[-1][1] # check if both solvers give the same solution quality
                lastTime1 = objs1[-1][0]
                lastTime2 = objs2[-1][0]
                scores['incomplete'] = (lastTime2/(lastTime1+lastTime2), lastTime1/(lastTime1+lastTime2))
    assert len(scores['complete'])==2
    assert len(scores['incomplete'])==2
    return scores


# check if a run r is as good as the best run on the same instance. Assumption: rB is never worse than r, and that the instance is solved by rB (decision problems) or at least a solution is found by rB (optimisation)
def close_to_best(r, rB, deltaTime=0.1):
    assert rB.status in ['S','C']
    # decision problems
    if r.problemType=='SAT':     
        assert rB.status in ['S','C']
        # instance is not solved by r
        if r.status not in ['S','C']:
            return False        
        # instance is solved by r
        assert r.totalTime >= rB.totalTime
        return (r.totalTime - rB.totalTime) <= deltaTime
    # optimisation problems
    if r.status != rB.status:
        assert r.status != 'C'
        return False
    # both r and rB solve the instance to completion
    if r.status == 'C':
        assert r.totalTime >= rB.totalTime
        return (r.totalTime - rB.totalTime) <= deltaTime    
    # both r and rB don't solve the instance to completion    
    if r.problemType=='MIN':
        if r.lastObjValue == rB.lastObjValue:
            assert r.lastObjTime >= rB.lastObjTime
            return (r.lastObjTime - rB.lastObjTime) <= deltaTime
        else:
            assert r.lastObjValue > rB.lastObjValue
            return False
    else:
        #print(r.problemType)
        #print(r)
        assert r.problemType == 'MAX'
        if r.lastObjValue == rB.lastObjValue:
            assert r.lastObjTime >= rB.lastObjTime
            return (r.lastObjTime - rB.lastObjTime) <= deltaTime
        else:
            assert r.lastObjValue < rB.lastObjValue
            return False   
        
        
def load_detail_results(path: str):
    t = pd.read_csv(path)
    t['solverAndTrack'] = t['solver']
    t['solver'] = ['-'.join(s.split('-')[:-1]) for s in t.solverAndTrack]
    t['track'] = [s.split('-')[-1] for s in t.solverAndTrack]
    t['isBest'] = False    
    t['rank'] = -1
    t = t.sort_values(by=['year','track','problem','instance','solver'])    
    
    # cap total time to time limit
    t.loc[(t.year<=2014) & (t.totalTime>900),'totalTime']=900
    t.loc[(t.year>2014) & (t.totalTime>1200),'totalTime']=1200

    # conjure doesn't like - and . and space and /
    for col in ['solver','problem','instance']:
        t[col] = [s.replace(' ','_') for s in t[col]]
        t[col] = [s.replace('-','_') for s in t[col]]
        t[col] = [s.replace('.','_dot_') for s in t[col]]
        t[col] = [s.replace('/','_') for s in t[col]]
        
    # promote results from fd -> free -> par -> open
    tracks = ['fd','free','par','open']
    t['track'] = pd.Categorical(t['track'], tracks)
    t = t.sort_values(by=['year','track','problem','instance','solver'])
    
    for year in t.year.unique():
        #display(t[t.year==year].groupby(['year','track']).solver.nunique())  
        for i in range(len(tracks)-1):
            fromTrack = tracks[i]
            for toTrack in tracks[(i+1):]:                     
                tf = t[(t.year==year) & (t.track==fromTrack)]
                tt = t[(t.year==year) & (t.track==toTrack)]
                promotedSolvers = set(tf.solver.unique()) - set(tt.solver.unique())
                print(f"year {year}: promote results from {fromTrack} to {toTrack}: {len(promotedSolvers)} solvers promoted")
                for solver in promotedSolvers:
                    t1 = tf[tf.solver==solver]
                    t1['track'] = toTrack
                    assert len(tf.track.unique())==1 # make sure we don't override the original table
                    t = pd.concat([t,t1], axis=0)                
        #display(t[t.year==year].groupby(['year','track']).solver.nunique())
        
    t = t.sort_values(by=['year','track','problem','instance','solver'])
    t['id'] = [i for i in range(len(t.index))]
    t = t.reset_index(drop=True)
        
    return t


def rank_solvers_per_instance(t: pd.DataFrame):
    """
    For each instance:
        mark the best run (isBest) per (year,track)
        rank all solvers according to their results
    """
    def mzn_comparator(id1, id2):
        r1 = t.loc[t.id==id1].iloc[0,:]                               
        r2 = t.loc[t.id==id2].iloc[0,:]    
        score = calculate_minizinc_borda_scores(r1.status, r2.status, r1.totalTime, r2.totalTime, r1.problemType, 
                                        [(r1.lastObjTime, r1.lastObjValue)], [(r2.lastObjTime, r2.lastObjValue)],
                                        True)['complete']
        if score[0] > score[1]:
            return -1
        if score[0] == score[1]:
            return 0
        return 1

    lsUnsolved = []
    n = 0
    curYear = None # for logging message
    curTrack = None # for logging message    
    curnInsts = 1 # for logging message
    t = t.sort_values(by=['year','track','problem','instance'])
    #t = t[(t.year==2013) & (t.track=='free')] # DEBUG
    for _, (yr,prob,inst,tk) in t[['year','problem','instance','track']].drop_duplicates().iterrows():
        if (yr != curYear) or (tk!=curTrack):
            print(f"\nProcessing year {yr} and track {tk}")
            curYear = yr
            curTrack = tk            
            curnInsts = 1        
        print(f"{curnInsts} instances processed",end='\r')
        t1 = t[(t.year==yr) & (t.problem==prob) & (t.instance==inst) & (t.track==tk)]
        # find the best run
        bestId = 0
        rBest = t1.iloc[bestId,:]
        for i in range(1,len(t1.index)):            
            ri = t1.iloc[i,:]
            scores = calculate_minizinc_borda_scores(rBest.status, ri.status, rBest.totalTime, ri.totalTime, rBest.problemType, [(rBest.lastObjTime, rBest.lastObjValue)], [(ri.lastObjTime, ri.lastObjValue)], True)
            sBest = scores['complete'][0]
            si = scores['complete'][1]
            if si>sBest:
                bestId = i
                rBest = t1.iloc[i,:]      
        # instance is not solved by any runs
        if rBest.status not in ['S','C']:
            lsUnsolved.append((yr,prob,inst,tk))
        else:        
            # mark the best run
            rowIndex = t1.iloc[bestId,:].id
            t.loc[t.id==rowIndex,'isBest'] = True
            # find runs that are close to the best runs             
            for i in range(len(t1.index)): 
                ri = t1.iloc[i,:]            
                #display(pd.concat([ri,rBest], axis=1).T) # for DEBUG only                
                if close_to_best(ri, rBest):
                    t.loc[t.id==t1.iloc[i,:].id,'isBest'] = True
            # sort all runs
            lsIds = t1['id'].tolist()        
            lsIds = sorted(lsIds, key=cmp_to_key(mzn_comparator))
            curRank = 1
            preId = None
            for cid in lsIds:                
                # equal runs should be ranked the same                
                if preId is not None:                    
                    v = mzn_comparator(preId, cid)                    
                    assert v<=0
                    if v<0:
                        curRank += 1
                t.loc[t.id==cid,'rank'] = curRank
                preId = cid
            #display(t.loc[t.id.isin(lsIds)].sort_values(by='rank'))    
            # DEBUG only
#             if (yr==2013) & (tk=='free') & (prob=='celar') & (inst=='CELAR6_SUB2'):
#                 print(rBest)
#                 print(t[(t.year==2013) & (t.track=='free') & (t.problem=='celar') & (t.instance=='CELAR6_SUB2')])
#             if len(t[(t.year==2013) & (t.track=='free') & (t.problem=='celar') & (t.instance=='CELAR6_SUB2') & (t.isBest)].index)>1:
#                 print("ERROR HERE")
#                 print(ri)
#                 print(rBest)
#                 sys.exit(1)

        curnInsts += 1

    print('')
    print(f"#unsolved instances: {len(lsUnsolved)}")    

    # remove unsolved instances
    for (yr,prob,inst,tk) in lsUnsolved:
        #display(t[(t.year==year) & (t.problem==prob) & (t.instance==inst) & (t.track==tk)])
        t.drop(t[(t.year==yr) & (t.problem==prob) & (t.instance==inst) & (t.track==tk)].index, inplace=True)
        #print(f"remove {yr} {prob} {inst} {tk})
        #display(t[(t.year==year) & (t.problem==prob) & (t.instance==inst) & (t.track==tk)])
        
    t = t.sort_values(by=['year','track','problem','instance','rank'])
        
    return t, lsUnsolved


def write_best_results_to_essence(t, year, track, outFile):    
    t1 = t[(t.year==year) & (t.track==track)]

    lsLines = []
    lsLines.append("letting instances be new type enum {" + ', '.join([f"{prob}__{inst}" for _, (prob, inst) in t1[['problem','instance']].drop_duplicates().iterrows()])+ "}") 
    lsLines.append("letting algorithms be new type enum {" + ', '.join(t1.solver.unique())+ "}")

    lsLines.append("letting isBest be function (")
    for alg in t1.solver.unique():
        s = '\t\t' + alg + " --> {"
        t2 = t1[(t1.solver==alg) & t1.isBest]  
        s += ', '.join([f"{prob}__{inst}" for _, (prob, inst) in t2[['problem','instance']].drop_duplicates().iterrows()])
        s += "}, "
        lsLines.append(s)
    lsLines.append("\t)")

    nSolvers = len(t1.solver.unique())

    lsLines.append(f"letting portfolioSize be {nSolvers}")

    with open(outFile, "wt") as f:
        f.write('\n'.join(lsLines))
        
    return nSolvers


def run_cmd(cmd, printOutput=False, outFile=None):
    p = subprocess.run(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    output = p.stdout.decode('utf-8')
    if outFile is not None:
        with open(outFile,'wt') as f:
            f.write(output)
    if printOutput:
        print(output)
    return output, p.returncode


def find_smallest_vbs_portfolios_with_conjure(paramFile):
    optimModel = "model"
    decisionModel = "model-decision"
    cjOutputDir = "./conjure-output/"

    # find VBS minimum portfolio size by solving the optimisation version first
    if os.path.isdir(cjOutputDir):
        shutil.rmtree(cjOutputDir)
    cmd = f"conjure solve {optimModel}.essence {paramFile} --output-format=json"
    print(cmd)
    output, returnCode = run_cmd(cmd)
    if returnCode!=0:
        print(f"ERROR: command '{cmd}' unsolvable")
        return None, None
    paramBaseName = os.path.basename(paramFile).replace(".param","")
    solutionFile = f"{cjOutputDir}/model000001-{paramBaseName}-solution000001.solution.json"
    with open(solutionFile, "rt") as f:
        rs = json.load(f)
    portfolioSize = len(rs['portfolio'])
    #print(f"Minimum VBS portfolio size: {portfolioSize}")

    ## now find all minimum-sized VBS portfolios

    # we first replace portfolioSize in paramFile with the newly found value
    with open(paramFile, "rt") as f:
        lsLines = [s.replace("\n","") for s in f.readlines()]
        for i in range(len(lsLines)):
            if "letting portfolioSize" in lsLines[i]:
                lsLines[i] = f"letting portfolioSize be {portfolioSize}"
                break
    with open(paramFile, "wt") as f:
        f.write('\n'.join(lsLines))

    # we now find all minimum-sized VBS portfolios by solving the decision version
    vbsPortfolios = []
    if os.path.isdir(cjOutputDir):
        shutil.rmtree(cjOutputDir)
    cmd = f"conjure solve {decisionModel}.essence {paramFile} --output-format=json --number-of-solution=all --output-format=json"
    print(cmd)
    output, returnCode = run_cmd(cmd)
    if returnCode!=0:
        print(f"ERROR: command '{cmd}' unsolvable")
        return None, None
    solutionFiles = glob.glob(cjOutputDir + "*.json")
    for solFile in solutionFiles:    
        with open(solFile, "rt") as f:
            rs = json.load(f)
            vbsPortfolios.append(rs['portfolio'])

    return vbsPortfolios, portfolioSize
    

def compare_portfolio_vs_vbs(portfolio, vbs, data):
    lsVBSScores = []
    lsScores = []
    for _,(prob, inst) in data[['problem','instance']].drop_duplicates().iterrows():    
        t2 = data[(data.problem==prob) & (data.instance==inst)]            
        t2 = t2.sort_values(by='rank')                    
        rBest = t2.iloc[0,:] # VBS                
        ri = t2[t2.solver.isin(portfolio)].iloc[0,:] # best of portfolio
        (sVBS, sNew) = calculate_minizinc_borda_scores(rBest.status, ri.status, rBest.totalTime, ri.totalTime, rBest.problemType, [(rBest.lastObjTime, rBest.lastObjValue)], [(ri.lastObjTime, ri.lastObjValue)], True)['complete']        
        assert sVBS >= sNew, f"ERROR: sVBS: {sVBS:.5f}, sNew: {sNew:.5f}"
        lsVBSScores.append(sVBS)
        lsScores.append(sNew)
    vbsTotalScores = sum(lsVBSScores)
    newTotalScores = sum(lsScores)
    return newTotalScores, vbsTotalScores
    
    
def filter_out_non_participants(t):
    with open("participants.json", "rt") as f:
        p = json.load(f)
    rs = None
    for year, solvers in p.items():        
        t1 = t[(t.year==int(year)) & (t.solver.isin(solvers))]
        if rs is None:
            rs = t1
        else:
            rs = pd.concat([rs, t1], axis=0)
    return rs

# calculate shapley values of each solver in a porfolio
def calculate_shapley_values_of_a_portfolio(port, ts):
    ts['port'] = [set(ast.literal_eval(p)) for p in ts.port]
    
    port = set(port)
    portSize = len(port)    
    print(f"portfolio size: {portSize}")
    # calculate shapley value of each solver
    rs = {}
    for solver in port: 
        print(f"Calculate Shapley value for {solver}")
        sv = 0 # shapley value
        pExclude = port - set([solver])  # all solvers excluding solver
        for size in range(1,portSize): 
            subPorts = list(itertools.combinations(pExclude,size))
            n = 0
            # go through all subsets of pExclude
            for p in subPorts:                
                p = set(p)
                sp = ts[ts.port==p].ratio.tolist()[0] # score of p
                p1 = p.union(set([solver]))                    
                sp1 = ts[ts.port==p1].ratio.tolist()[0] # score of {p union solver}
                assert sp1 >= sp
                sv += sp1-sp                                    
                n += 1    
                print(f"size {size:2}, calculated: {n:5}", end="\r")                
        rs[solver] = sv
    return rs

def merge_tables(lsFiles, outFile):
    t = None
    for fn in lsFiles:
        t1 = pd.read_csv(fn)
        if t is None:
            t = t1
        else:
            t = pd.concat([t,t1], axis=0)
    t.to_csv(outFile, index=False)
    