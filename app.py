import dash
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from io import StringIO
import sys

# Import des données
cancer = pd.read_csv('data/Breast_cancer_subtypes_samples.csv',sep=';')
GSE21653 = pd.read_csv('data/expression_data_GSE21653_GSE21653_log_expression_266_samples_21887_genes.csv',sep=';')
Miller = pd.read_csv('data/expression_data_Miller-2005_Miller-2005_log_expression_251_samples_14145_genes.csv',sep=';')
Naderi_Caldas = pd.read_csv('data/expression_data_Naderi-Caldas-2007_Naderi-Caldas-2007_log_expression_242_samples_14366_genes.csv',sep=';')
E_MTAB_365 = pd.read_csv('data/expression_data_probreast_microarrays_E-MTAB-365_log_expression_1190_samples_23035_genes.csv',sep=';')
GSE25066 = pd.read_csv('data/expression_data_probreast_microarrays_GSE25066_log_expression_508_samples_13815_genes.csv',sep=';')
GSE42568 = pd.read_csv('data/expression_data_probreast_microarrays_GSE42568_log_expression_121_samples_23035_genes.csv',sep=';')
TCGA_BRCA = pd.read_csv('data/expression_data_tcga_brca_TCGA-BRCA_log_fpkm_1250_samples_42851_genes.csv',sep=';')
Yau = pd.read_csv('data/expression_data_Yau-2010_Yau-2010_log_expression_683_samples_8791_genes.csv',sep=';')
GSE21653_EG = pd.read_excel('data/EpiMed_experimental_grouping_2022.11.28_GSE21653.xlsx')
GSE25066_EG = pd.read_excel('data/EpiMed_experimental_grouping_2022.11.28_GSE25066.xlsx')
GSE42568_EG = pd.read_excel('data/EpiMed_experimental_grouping_2022.11.28_GSE42568.xlsx')
TCGA_BRCA_EG = pd.read_excel('data/EpiMed_experimental_grouping_2022.11.28_TCGA-BRCA.xlsx')
Naderi_Caldas_EG = pd.read_excel('data/EpiMed_experimental_grouping_2022.12.01_Naderi-Caldas-2007.xlsx')
Yau_EG = pd.read_excel('data/EpiMed_experimental_grouping_2022.12.01_Yau-2010.xlsx')
Miller_EG = pd.read_excel('data/EpiMed_experimental_grouping_2022.12.02_Miller-2005.xlsx')
E_MTAB_365_EG = pd.read_excel('data/EpiMed_experimental_grouping_2022.11.28_E-MTAB-365.xlsx')

# Fonctions
# Création de la fonction gene_expression
# Elle renvoie un tableau qui donne (pour chaque échantillon d'une jeu de données) le type de cancer ainsi que l'expression du gène donné en entrée
def gene_expression(dataset,gene,dataset2):
    TDC = cancer[cancer["Dataset"] == dataset][["Luminal-A","Luminal-B","HER2-enriched","Basal-like"]]
    ech = cancer[cancer["Dataset"] == dataset]['Sample']
    type_cancer = TDC.apply(lambda row: ''.join(row.keys()[row == 1]), axis=1)
    type_cancer = pd.DataFrame(type_cancer)
    type_cancer = type_cancer.T
    type_cancer.columns = ech.to_list()
    
    dataset2.gene_symbol = dataset2.gene_symbol.convert_dtypes()
    EG = dataset2[dataset2.gene_symbol == gene][cancer[cancer["Dataset"] == dataset]['Sample']]
    df = pd.concat([type_cancer,EG],axis=0)
    df.index = ['type', 'value']
    df = df.T
    df.value = pd.to_numeric(df.value)
    df['type'] = df['type'].replace({'': 'Non-tumour'})
    return df # sns.boxplot(x=df["type"],y=df["value"],orient = "v")
gene_expression('Naderi-Caldas-2007',"BOP1",globals()['Naderi_Caldas']) # -> renvoie aucun type
gene_expression('TCGA-BRCA',"EZH2",globals()['TCGA_BRCA']) # Lui il marche

def get_logfc(dataset,gene,dataset2):
    df = gene_expression(dataset,gene,dataset2)
    df2 = df.groupby("type")["value"].mean()
    result = pd.DataFrame({'gene': [gene] * len(df2.index), 'dataset': [dataset] * len(df2.index), 'type': df2.index, 'LOGfc': [None] * len(df2.index),
                           'fc': [None] * len(df2.index), 'F': [None]*len(df2.index), 'pvalue': [None] * len(df2.index),
                           'FDR': [None] * len(df2.index)})
    for group in result.index:
        other_groups = result.index.difference([group])
        average_value = df2[other_groups].mean()
        fc = df2[group].mean() - average_value
        F, p_value = f_oneway(df['value'][df.type == result['type'][group]], df['value'][df.type != result['type'][group]])
        p_valeurs_corrigees = multipletests(p_value, method='fdr_bh')[1][0]
        result.at[group, 'LOGfc'] = fc
        result.at[group, 'fc'] = 2**fc
        result.at[group, 'F'] = F
        result.at[group, 'pvalue'] = p_value
        result.at[group, 'FDR'] = p_valeurs_corrigees
    return result

def get_survie(dataset,dataset2, gene):
    data = globals()[dataset]
    data2 = globals()[dataset2]
    expression = data[data.gene_symbol == gene]
    expression.dropna(axis=1)
    survie = data2[["id_sample","os_months","os_censor","dfs_months","dfs_censor"]]
    # Ce if permet d'arrêter la fonction si les données sont manquantes
    if(survie.os_months.value_counts().shape[0] == 0 or survie.os_censor.value_counts().shape[0] == 0):
        return "Les données de survie ne sont pas disponible"
    survie = survie.dropna()
    expression = expression[data2[["id_sample","os_months","os_censor","dfs_months","dfs_censor"]]['id_sample'].values].dropna(axis=1)
    expression = expression[survie.id_sample].T
    expression.columns = ["value"]
    survie = pd.concat([survie.reset_index(drop=True),expression.reset_index(drop=True)],axis=1)
    mediane = np.percentile(survie["value"],50)
    survie[survie["value"] > mediane]["os_months"]

    mediane = np.percentile(survie["value"],50)
    kmf1 = KaplanMeierFitter()
    kmf2 = KaplanMeierFitter()
    high = survie[survie["value"] > mediane]
    low = survie[survie["value"] < mediane]
    kmf1.fit(durations=high["os_months"], event_observed=high['os_censor'])
    kmf2.fit(durations=low["os_months"], event_observed=low['os_censor'])
    plt.figure(figsize=(10, 6))
    kmf2.plot_survival_function(label=f'High n = {high.shape[0]}',color='red',ci_alpha=0)
    kmf1.plot_survival_function(label=f'Low n = {low.shape[0]}',color='blue',ci_alpha=0)
    plt.xlim(0,min(len(low),len(high)))
    plt.title(f'{gene} - {dataset}') 
    # plt.title(f'EZH2 - E-MTAB-365\ncox p-value : {res['p'].values}\nlogrank p-value : {res_logr.summary['p']}')
    plt.xlabel('Temps (en mois)')
    plt.ylabel('Probabilité de survie')
    plt.legend()
    plt.show()

# Import des 35 KDM
with open("genes_interets/KDM.txt", 'r') as f:
    lignes = f.readlines()
id = eval(lignes[0].strip())
nom_gene = eval(lignes[1].strip())
KMT = pd.DataFrame({'id': id, 'nom_gene': nom_gene}) # , 'type': np.repeat("KMT",len(id))


# Import des 109 KMT
with open("genes_interets/KMT.txt", 'r') as f:
    lignes = f.readlines()
id = eval(lignes[0].strip())
nom_gene = eval(lignes[1].strip())
KDM = pd.DataFrame({'id': id, 'nom_gene': nom_gene}) # , 'type': np.repeat("KDM",len(id))

# Import des 461 KMB
with open("genes_interets/KMB.txt", 'r') as f:
    lignes = f.readlines()
id = eval(lignes[0].strip())
nom_gene = eval(lignes[1].strip())
KMB = pd.DataFrame({'id': id, 'nom_gene': nom_gene}) # , 'type': np.repeat("KMB",len(id))

GI = pd.concat([KMT,KDM,KMB])

GSE21653 = GSE21653[GSE21653["id_gene"].isin(GI.id.values)]
Miller = Miller[Miller["id_gene"].isin(GI.id.values)]
Naderi_Caldas = Naderi_Caldas[Naderi_Caldas["id_gene"].isin(GI.id.values)]
E_MTAB_365 = E_MTAB_365[E_MTAB_365["id_gene"].isin(GI.id.values)]
GSE25066 = GSE25066[GSE25066["id_gene"].isin(GI.id.values)]
GSE42568 = GSE42568[GSE42568["id_gene"].isin(GI.id.values)]
TCGA_BRCA = TCGA_BRCA[TCGA_BRCA["id_gene"].isin(GI.id.values)]
Yau = Yau[Yau["id_gene"].isin(GI.id.values)]

GI.columns = ["id_gene", "nom_gene"]

liste_dataset = ["GSE21653","Miller","Naderi_Caldas","E_MTAB_365",
                 "GSE25066","GSE42568","TCGA_BRCA" ,"Yau"]

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children="Application pour l'analyse de survie", style={'textAlign':'center'}),
    dcc.Dropdown(liste_dataset, 'GSE21653', id='dataset'),
    # dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
    # dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
    dcc.Graph(id='graph')
])

@callback(
    Output('graph', 'figure'),
    Input('dataset', 'value')
)
def update_graph(value):
    return get_survie('value','value'+"_EG","EZH2")

if __name__ == '__main__':
    app.run(debug=True)