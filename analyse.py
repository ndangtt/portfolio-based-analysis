import pandas as pd
import numpy as np
import sys
import os
import shutil
from pprint import pprint
import os
from utils import load_detail_results, rank_solvers_per_instance, write_best_results_to_essence, find_smallest_vbs_portfolios_with_conjure, calculate_minizinc_borda_scores, compare_portfolio_vs_vbs, filter_out_non_participants, calculate_shapley_values_of_a_portfolio
import itertools
import fire
import ast
from ast import literal_eval
import json
import glob
sys.path.append(os.path.dirname(__file__))


def produce_ranking_per_instance(participantOnly:bool=False):
    t = load_detail_results('data/detailed_results.csv')

    if participantOnly is False:
        t, lsUnsolved = rank_solvers_per_instance(t)

        t.to_csv("results/full_results_with_ranks.csv",index=False)
        with open("unsolved.txt","wt") as f:
            for r in lsUnsolved:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")

    else:
        t = filter_out_non_participants(t)

        t, lsUnsolved = rank_solvers_per_instance(t)

        t.to_csv("results/full_results_with_ranks_participants_only.csv",index=False)
        with open("results/unsolved_participant_only.txt","wt") as f:
            for r in lsUnsolved:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")    


def find_vbs_portfolios(participantOnly:bool=False):
    if os.path.isdir("tmp") is False:
        os.mkdir("tmp")
        
    if participantOnly:
        t = pd.read_csv("results/full_results_with_ranks_participants_only.csv")
        outFile = "results/smallest_vbs_portfolios_participants_only.csv"
    else:
        t = pd.read_csv("results/full_results_with_ranks.csv")
        outFile = "results/smallest_vbs_portfolios.csv"
        r
    allVBSPortfolios = []

    for _, (year,track) in t[['year','track']].drop_duplicates().iterrows():
        print(f"\nSearch for minimum portfolios for year {year} and track {track}...")
        paramFile = f"tmp/{year}-{track}.param"
        nSolvers = write_best_results_to_essence(t, year, track, paramFile)

        vbsPortfolios, vbsPortfolioSize = find_smallest_vbs_portfolios_with_conjure(paramFile)
        if vbsPortfolios is None:
            print("ERROR")
        
        # TODO: uncomment those, but we should wait for a few seconds first. It seems that those commands can sometime run before conjure is done with solution parsing...
        #os.remove(paramFile)
        #os.remove(f"model-{year}-{track}.solution")
        #os.remove(f"model-decision-{year}-{track}.solution")

        print(f"#solvers: {nSolvers}, minimum VBS portfolio size: {vbsPortfolioSize}, #portfolios: {len(vbsPortfolios)}")
        #print(vbsPortfolios)
        allVBSPortfolios.append({'year':year, 'track':track, 'nSolvers': nSolvers, 'vbsSize': vbsPortfolioSize, 'nVBSPortfolios': len(vbsPortfolios), 'vbsPortfolios': vbsPortfolios})
        
    tAllVBS = pd.DataFrame(allVBSPortfolios)
    tAllVBS.to_csv(outFile, index=False)
    

def compare_vbs_participants_only_vs_vbs_all():
    t = pd.read_csv("results/full_results_with_ranks.csv")
    tvbsAll = pd.read_csv("results/smallest_vbs_portfolios.csv")
    tvbsP = pd.read_csv("results/smallest_vbs_portfolios_participants_only.csv")
    #display(tvbsP)
    rs = []
    for _, (year, track) in t[['year','track']].drop_duplicates().iterrows():
        #if track=="open": 
        #    continue
        vbsAll = ast.literal_eval(tvbsAll[(tvbsAll.year==year) & (tvbsAll.track==track)]['vbsPortfolios'].tolist()[0])[0]
        vbsP = ast.literal_eval(tvbsP[(tvbsP.year==year) & (tvbsP.track==track)]['vbsPortfolios'].tolist()[0])[0]
        data = t[(t.year==year) & (t.track==track)]
        pScore, allScore = compare_portfolio_vs_vbs(vbsP, vbsAll, data)   
        print(f"year: {year}, track: {track}, participants only: {pScore:.3f}, all: {allScore:.3f}, ratio: {pScore/allScore:.3f}")
        rs.append({'year': year, 'track': track, 'participantVBSScore': pScore, 'allVBSScore': allScore, 'ratio': pScore/allScore})
    rs = pd.DataFrame(rs)
    rs.to_csv("results/all_vs_participants_only_scores.csv", index=False)
    

def calculate_portfolio_scores(year, track, participantOnly:bool=False):
    if participantOnly:
        tAllVBS = pd.read_csv("results/smallest_vbs_portfolios_participants_only.csv")
        t = pd.read_csv("results/full_results_with_ranks_participants_only.csv")  
        outFile = f"results/portfolioScores-{year}-{track}-participants_only.csv"
    else:
        tAllVBS = pd.read_csv("results/smallest_vbs_portfolios.csv")
        t = pd.read_csv("results/full_results_with_ranks.csv")    
        outFile = f"results/portfolioScores-{year}-{track}.csv"
        
    tvbs = tAllVBS[(tAllVBS.year==year) & (tAllVBS.track==track)]    
    assert len(tvbs.index)==1
    data = t[(t.year==year) & (t.track==track)]
    # save scores here
    ts = []    
    # list of calculated portfolios
    done = []
    # all vbs portfolios
    lsVBSPorts = ast.literal_eval(tvbs.vbsPortfolios.to_list()[0])
    # calculate scores
    k = 1
    for vbsPort in lsVBSPorts:
        print(f"Calculate portfolio scores for year {year} and track {track}, VBS portfolio {vbsPort}")
        for size in range(len(vbsPort), 0, -1):            
        #for size in [1]: # TODO: REMOVE
            ports = list(itertools.combinations(vbsPort,size)) # all subsets of this size            
            nPorts = 0
            for p in ports:
                if set(p) in done:                
                    continue       
                #print(p) # TODO: REMOVE
                pScore, vbsScore = compare_portfolio_vs_vbs(p, vbsPort, data)                
                ts.append({'year':year, 'track': track, 'portSize': size, 'port': p, 'portScore': pScore, 'vbsScore': vbsScore, 'ratio': pScore/vbsScore})                
                #print(f"Score for {str(p)}: {pScore/vbsScore:.2f}")
                nPorts += 1
                print(f"size: {size}, calculated: {nPorts}/{len(ports)}", end="\r")
                done.append(set(p))
            print("")
        k+=1
    ts = pd.DataFrame(ts)
    ts.to_csv(outFile, index=False)     
    
    
def combine_portfolio_scores():
    lsFiles = glob.glob("results/portfolioScores-*-participants_only.csv")
    ts = None
    for fn in lsFiles:
        t1 = pd.read_csv(fn)
        if ts is None:
            ts = t1
        else:
            ts = pd.concat([ts,t1],axis=0)
    ts = ts.sort_values(by=['year','track','portSize'])
    ts.to_csv("results/portfolioScores-participants_only.csv", index=False)

    lsFiles = glob.glob("results/portfolioScores-*-free.csv")
    ts = None
    for fn in lsFiles:
        t1 = pd.read_csv(fn)
        if ts is None:
            ts = t1
        else:
            ts = pd.concat([ts,t1],axis=0)
    ts = ts.sort_values(by=['year','track','portSize'])
    ts.to_csv("results/portfolioScores.csv", index=False)
    
    
def calculate_scores_of_best_portfolios():
    """
    Combine best porffolio of each size for:
        - all solvers
        - participants only
    Then calculate their scores vs VBS (of all solvers)
    """
    t1 = pd.read_csv("results/portfolioScores.csv")
    t2 = pd.read_csv("results/portfolioScores-participants_only.csv")
    tvbs = pd.read_csv("results/smallest_vbs_portfolios.csv")
    t = pd.read_csv("results/full_results_with_ranks.csv")
    
    t1Best = t1.groupby(['year','track','portSize']).ratio.max().reset_index().merge(t1[['year','track','ratio','portSize','port']],how='left',on=['year','track','ratio','portSize'])    
    t1Best['set'] = 'all'
    
    # for data in portfolioScores-participants_only, we need to recalculate the score using VBS of all solvers
    t2Best = t2.groupby(['year','track','portSize']).ratio.max().reset_index().merge(t2[['year','track','ratio','portSize','port']],how='left',on=['year','track','ratio','portSize'])    
    t2BestNew = []    
    for _,row in t2Best.iterrows():                  
        vbs = ast.literal_eval(tvbs[(tvbs.year==row.year) & (tvbs.track==row.track)].vbsPortfolios.tolist()[0])[0]        
        data = t[(t.year==row.year) & (t.track==row.track)]
        pScore, vbsScore = compare_portfolio_vs_vbs(ast.literal_eval(row.port), vbs, data)
        ratio = pScore / vbsScore
        assert ratio <= row.ratio
        row['ratio'] = ratio
        t2BestNew.append(row.to_dict())
    t2Best = pd.DataFrame(t2BestNew)
    t2Best['set'] = 'participants-only'
    
    tAll = pd.concat([t1Best, t2Best], axis=0)
    tAll.to_csv('results/best_portfolio_scores_combined.csv', index=False)    
    
    
def calculate_shapley_values(year, track, participantOnly=False):    
    if participantOnly:
        t = pd.read_csv("results/smallest_vbs_portfolios_participants_only.csv")    
        ts = pd.read_csv('results/portfolioScores-participants_only.csv')
        outFile = f"results/shapley-{year}-{track}-participants_only.json"
    else:
        t = pd.read_csv("results/smallest_vbs_portfolios.csv")    
        ts = pd.read_csv('results/portfolioScores.csv')
        outFile = f"results/shapley-{year}-{track}.json"
              
    port = ast.literal_eval(t[(t.year==year) & (t.track==track)].vbsPortfolios.tolist()[0])[0]
            
    ts = ts[(ts.year==year) & (ts.track==track)]
    values = calculate_shapley_values_of_a_portfolio(port, ts)
    rs = {'year':year, 'track': track, 'port': port, 'shapley': values}
    with open(outFile, "wt") as f:
        json.dump(rs, f)
        
def calculate_borda_scores(year, track):
    t = pd.read_csv("results/full_results_with_ranks.csv")
    t = t[(t.year==year) & (t.track==track)]
    lsSolvers = t.solver.unique().tolist()
    rs = []
    print(f"Calculate borda score for year {year} and track {track}")
    for solver1 in lsSolvers:
        for solver2 in set(lsSolvers) - set([solver1]):
            for _,(prob,inst) in t[['problem','instance']].drop_duplicates().iterrows():
                r1 = t[(t.problem==prob) & (t.instance==inst) & (t.solver==solver1)].iloc[0,:]
                r2 = t[(t.problem==prob) & (t.instance==inst) & (t.solver==solver2)].iloc[0,:]              
                score = calculate_minizinc_borda_scores(r1.status, r2.status, r1.totalTime, r2.totalTime, r1.problemType, 
                                        [(r1.lastObjTime, r1.lastObjValue)], [(r2.lastObjTime, r2.lastObjValue)],
                                        False)['complete']
                rs.append({'year':year, 'track': track, 'problem':prob, 'instance': inst, 
                           'solver1': solver1, 'solver2': solver2, 'score1': score[0], 'score2': score[1]})
    rs = pd.DataFrame(rs)
    rs.to_csv(f"borda-{year}-{track}.csv", index=False)        
    
def combine_borda_and_shapley_values(participantsOnly=False):
    if participantsOnly:
        vbsFile = "results/smallest_vbs_portfolios_participants_only.csv"
        slFile = "results/shapley-participants_only.json"
        outFile = "results/combined_scores_participants_only.csv"
        with open("results/sparticipants.json", "r") as f:
            participants = json.load(f)
    else:
        vbsFile = "results/smallest_vbs_portfolios.csv"
        slFile = "results/shapley.json"
        outFile = "results/combined_scores.csv"
    
    # read borda scores and shapley values
    tb = pd.read_csv("results/borda.csv")
    with open(slFile,'rt') as f:
        shapley = [json.loads(s.replace('\n','')) for s in f.readlines()]

    # participants + non-participants
    t = pd.read_csv(vbsFile)
    t = t[t.track.isin(['free'])]
    rs = None
    for _,(year,track) in t[['year','track']].drop_duplicates().iterrows():
        #print(f"year {year}, track {track}")
        t2 = t[(t.year==year) & (t.track==track)]
        port = ast.literal_eval(t2.vbsPortfolios.tolist()[0])[0]

        # borda score of each solver within port    
        tbPort = tb[tb.solver1.isin(port) & tb.solver2.isin(port)]   
        tbPort = tbPort[(tbPort.year==year) & (tbPort.track==track)]
        nPairs = len(tbPort[['solver1','solver2']].drop_duplicates().index)
        tb1 = tbPort.groupby(['solver1']).score1.sum().reset_index().sort_values(by='solver1')
        tb2 = tbPort.groupby(['solver2']).score2.sum().reset_index().sort_values(by='solver2')
        tb1.rename({'solver1': 'solver', 'score1': 'bordaScore'}, axis=1, inplace=True)
        tb2.rename({'solver2': 'solver', 'score2': 'bordaScore'}, axis=1, inplace=True)
        tbScore = tb1
        tbScore['bordaScore'] = (tb1.bordaScore + tb2.bordaScore)/nPairs
        tbScore['bordaRank'] = tbScore['bordaScore'].rank(ascending=False)

        # borda score of each solver in the competition (based on all solvers)
        tbPort = tb[(tb.year==year) & (tb.track==track)]
        if participantsOnly:
            tbPort = tbPort[tbPort.solver1.isin(participants[str(year)])]
            tbPort = tbPort[tbPort.solver2.isin(participants[str(year)])]
        nPairs = len(tbPort[['solver1','solver2']].drop_duplicates().index)
        tb1 = tbPort.groupby(['solver1']).score1.sum().reset_index().sort_values(by='solver1')
        tb2 = tbPort.groupby(['solver2']).score2.sum().reset_index().sort_values(by='solver2')
        tb1.rename({'solver1': 'solver', 'score1': 'bordaScoreAll'}, axis=1, inplace=True)
        tb2.rename({'solver2': 'solver', 'score2': 'bordaScoreAll'}, axis=1, inplace=True)
        tb3 = tb1
        tb3['bordaScoreAll'] = (tb1.bordaScoreAll + tb2.bordaScoreAll)/nPairs
        tb3['bordaRankAll'] = tb3['bordaScoreAll'].rank(ascending=False)
        tb3 = tb3.merge(tbScore, on='solver', how='outer')    

        # shapley value of each solver within port
        nSubsets = 2**(len(port)-1)
        svals = [v['shapley'] for v in shapley if set(v['port']) == set(port)][0]
        tsScore = pd.DataFrame([{'solver': solver, 'shapleyValue': val/nSubsets*100} for solver, val in svals.items()])
        tsScore['shapleyRank'] = tsScore['shapleyValue'].rank(ascending=False)

        # combine borda and shapley into one table
        tScore = tb3.merge(tsScore, on='solver', how='outer')    
        tScore['year'] = year
        tScore['track'] = track

        # add to rs
        if rs is None:
            rs = tScore
        else:
            rs = pd.concat([rs, tScore], axis=0)
    rs.to_csv(outFile, index=False)

if __name__=="__main__":            
    produce_ranking_per_instance(True)
    find_vbs_portfolios(True)

    produce_ranking_per_instance(False)
    find_vbs_portfolios(False)
    
    compare_vbs_participants_only_vs_vbs_all()

    combine_portfolio_scores()

    for year in range(2013,2022):
        calculate_scores_of_best_portfolios(year, 'free', True)
        calculate_scores_of_best_portfolios(year, 'free', False)
        calculate_shapley_values(year,'free')
        calculate_borda_scores(year,'free')

    combine_borda_and_shapley_values(True)
    combine_borda_and_shapley_values(False)
    
