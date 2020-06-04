import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import goatools

# need the following to get GO term name from ID
from goatools.base import download_go_basic_obo
obo_fname = download_go_basic_obo()
from goatools.obo_parser import GODag
obodag = GODag("go-basic.obo")

# load data
protein_data = pd.read_csv('Data/protein_consensus.csv')
protein_data.columns = ['Gene'] + list(protein_data.columns)[1:] #renaming first column to Gene

#read in GO terms to gene mapping dataframe
geneGO_df = pd.read_csv('Data/gene_GO_df.csv') #this dataferame was obtained in R using biomaRt
geneGO_data = geneGO_df.loc[geneGO_df.ensembl_gene_id.isin(list(protein_data['Gene'])), :] #limiting to genes that are in data

# create master list of GO terms
master_GO_list = []
for i,row in geneGO_data.iterrows():
    GO_id = row['go_id']
    if GO_id not in master_GO_list:
        master_GO_list.append(GO_id)

# generate dictionary with GO terms as keys and list of genes as values
GO_dict = {}
#limit GO terms to ontologies with more than 10 genes measured
for GO in master_GO_list:
    GO_genes = list(geneGO_data.loc[geneGO_data.go_id==GO, 'ensembl_gene_id'].values)
    if len(GO_genes)>=10:
        GO_dict[GO] = GO_genes

#function for running gene-set enrichment analysis
def GSEA(data):
    
    print('Preprocessing data')
    # normalize by sample median
    norm_p_data = data #initializing df
    for i, col in enumerate(list(norm_p_data.columns)[1:]):
        median_col = np.median([x for x in norm_p_data[col].values if ~np.isnan(x)])
        norm_col = [x/median_col for x in norm_p_data[col].values]
        norm_p_data.loc[:,col] = norm_col
    
    # normalize relative to across-tissue median and impute missing values with median
    norm_pt_data = norm_p_data
    for i, row in norm_pt_data.iterrows():
        row_val = row.values[1:]
        median_row = np.median([x for x in row_val if ~np.isnan(x)]) 
        row_val = [x if ~np.isnan(x) else median_row for x in row_val]
        norm_row = [x/median_row for x in row_val]
        norm_pt_data.loc[i,(norm_pt_data.columns)[1:]] = norm_row
    
    print('GSEA')
    # GSEA
    tissues = norm_pt_data.columns[1:]
    #initializing dfs for results
    enrichment_pvals = pd.DataFrame(index=list(GO_dict.keys()), columns=tissues)
    enrichment_ks = pd.DataFrame(index=list(GO_dict.keys()), columns=tissues)

    # loop over GO terms and tissues, identify associated gene-set, compare distribution of abundance in tissue to distribution of abundance in all other tissues using KS-test, save KS statistic and p-values
    for k,v in GO_dict.items():
        for t in tissues:
            nt = [x for x in tissues if x!=t]
            tissue_data = list(norm_pt_data.loc[norm_pt_data.Gene.isin(v), t].values)
            nontissue_data = [list(norm_pt_data.loc[norm_pt_data.Gene.isin(v), x].values) for x in nt]
            nt_data = [x for y in nontissue_data for x in y] #flatten list
            ks_test = stats.ks_2samp(tissue_data, nt_data)
            enrichment_pvals.loc[k,t] = ks_test[1]
            enrichment_ks.loc[k,t] = ks_test[0]
    
    print('FDR')
    # obtain 5% FDR corrected p-values using benjamini hochburg method
    FDR_corrected_pvals = enrichment_pvals
    for i, col in enumerate(list(FDR_corrected_pvals.columns)):
        pvals = list(FDR_corrected_pvals[col].values)
        FDR_pvals = multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
        FDR_corrected_pvals.loc[:,col] = FDR_pvals
    
    #to find the GO terms that vary the most across all 13 tissues, rank by median KS statistic across the tissues
    enrichment_ks['Median'] = enrichment_ks.apply(lambda x: np.median(x), axis=1)
    enrichment_ks.sort_values('Median', ascending=False, inplace=True)
    top50 = list(enrichment_ks.index)[0:50] 
    top50_names = []
    for i in top50:
        top50_names.append(obodag[i].name)
    
    print('Generating heatmap')
    # generate heatmap of log2 tissue abundance ratios
    tissue_ratios = pd.DataFrame(index=top50_names, columns=tissues)
    for i,GO in enumerate(top50):
        v = GO_dict[GO]
        for t in tissues:
            tissue_data = list(norm_pt_data.loc[norm_pt_data.Gene.isin(v), t].values)
            tissue_ratios.loc[top50_names[i], t] = np.log2(np.mean(tissue_data))
    tissue_ratios = tissue_ratios.astype(float)
    fig, ax = plt.subplots(figsize=(10,12))
    sns.heatmap(tissue_ratios, cmap=sns.color_palette('RdBu_r', 500))
    plt.tight_layout()
    plt.savefig('Results/GSEA_heatmap.pdf')

GSEA(protein_data)
