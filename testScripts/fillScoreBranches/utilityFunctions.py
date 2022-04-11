### Reading the command line
from argparse import ArgumentParser
import sys
from colorama import init, Fore
init(autoreset = True)


def SelectEvents(dataFrame, channel, analysis):
    ### Selecting events according to type of analysis and channel
    selectionMergedGGF = 'Pass_MergHP_GGF_ZZ_Tag_SR == True or Pass_MergHP_GGF_ZZ_Untag_SR == True or Pass_MergHP_GGF_WZ_SR == True or Pass_MergLP_GGF_ZZ_Tag_SR == True or Pass_MergLP_GGF_ZZ_Untag_SR == True or Pass_MergLP_GGF_WZ_SR == True or Pass_MergHP_GGF_ZZ_Tag_ZCR == True or Pass_MergHP_GGF_ZZ_Untag_ZCR == True or Pass_MergHP_GGF_WZ_ZCR == True or Pass_MergLP_GGF_ZZ_Tag_ZCR == True or Pass_MergLP_GGF_ZZ_Untag_ZCR == True or Pass_MergLP_GGF_WZ_ZCR == True'# or Pass_MergHP_GGF_ZZ_Untag_TCR == True or Pass_MergHP_GGF_ZZ_Tag_TCR == True or Pass_MergHP_GGF_WZ_TCR == True or Pass_MergLP_GGF_ZZ_Untag_TCR == True or Pass_MergLP_GGF_ZZ_Tag_TCR == True or Pass_MergLP_GGF_WZ_TCR == True'
    selectionMergedGGFZZLPuntagSR = 'Pass_MergLP_GGF_ZZ_Untag_SR == True and Pass_MergHP_GGF_ZZ_Tag_SR == False and Pass_MergHP_GGF_ZZ_Untag_SR == False and Pass_MergHP_GGF_WZ_SR == False and Pass_MergLP_GGF_ZZ_Tag_SR == False and Pass_MergHP_GGF_ZZ_Tag_ZCR == False and Pass_MergHP_GGF_WZ_ZCR == False and Pass_MergHP_GGF_ZZ_Untag_ZCR == False and Pass_MergLP_GGF_ZZ_Tag_ZCR == False and Pass_MergLP_GGF_ZZ_Untag_ZCR == False and Pass_MergLP_GGF_WZ_SR == False and Pass_MergLP_GGF_WZ_ZCR == False'
    selectionMergedVBF = 'Pass_MergHP_VBF_WZ_SR == True or Pass_MergHP_VBF_ZZ_SR == True or Pass_MergLP_VBF_WZ_SR == True or Pass_MergLP_VBF_ZZ_SR == True or Pass_MergHP_VBF_WZ_ZCR == True or Pass_MergHP_VBF_ZZ_ZCR == True or Pass_MergLP_VBF_WZ_ZCR == True or Pass_MergLP_VBF_ZZ_ZCR == True'# or Pass_MergHP_VBF_WZ_TCR == True or Pass_MergHP_VBF_ZZ_TCR == True or Pass_MergLP_VBF_WZ_TCR == True or Pass_MergLP_VBF_ZZ_TCR == True'
    selectionResolvedGGF = 'Pass_Res_GGF_WZ_SR == True or Pass_Res_GGF_ZZ_Tag_SR == True or Pass_Res_GGF_ZZ_Untag_SR == True or Pass_Res_GGF_WZ_ZCR == True or Pass_Res_GGF_ZZ_Tag_ZCR == True or Pass_Res_GGF_ZZ_Untag_ZCR == True'# or Pass_Res_GGF_WZ_TCR == True or Pass_Res_GGF_ZZ_Tag_TCR == True or Pass_Res_GGF_ZZ_Untag_TCR == True'
    selectionResolvedVBF = 'Pass_Res_VBF_WZ_SR == True or Pass_Res_VBF_ZZ_SR == True or Pass_Res_VBF_WZ_ZCR == True or Pass_Res_VBF_ZZ_ZCR == True'# or Pass_Res_VBF_WZ_TCR == True or Pass_Res_VBF_ZZ_TCR == True'

    selectionAll = selectionResolvedVBF + ' or ' + selectionResolvedGGF + ' or ' + selectionMergedVBF + ' or ' + selectionMergedGGF

    if channel == 'all':
        selection = selectionAll
    elif channel == 'ggF':
        dataFrame = dataFrame.query('Pass_isVBF == False')
        if analysis == 'merged':
            selection = selectionMergedGGF
            #selection = selectionMergedGGFZZLPuntagSR            
        elif analysis == 'resolved':
            selection = selectionResolvedGGF
    elif channel == 'VBF':
        dataFrame = dataFrame.query('Pass_isVBF == True')
        if analysis == 'merged':
            selection = selectionMergedVBF
        elif analysis == 'resolved':
            selection = selectionResolvedVBF
    dataFrame = dataFrame.query(selection)

    return dataFrame

