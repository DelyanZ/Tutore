def plot_HR(soustype, nb, typesurvie = None):
    if typesurvie == 'DFS':
        best_final = all_survie2[all_survie2.typesurvie == "DFS"].sort_values(by = "HR", ascending=False)[8:13291]
        best_final = best_final[best_final.soustype == soustype]
        # res = res_DFS.sort_values(by = 'dataset', ascending = False).reset_index()
        res = best_final.groupby(["gene", "conclusion"])["dataset"].nunique().reset_index().sort_values(by='dataset',ascending=False)
    if typesurvie == "OS":
        best_final = all_survie2[all_survie2.typesurvie == "OS"].sort_values(by = "HR", ascending=False)[47:8392]
        best_final = best_final[best_final.soustype == soustype]
        # res = res_OS.sort_values(by = 'dataset', ascending = False).reset_index()
        res = best_final.groupby(["gene", "conclusion"])["dataset"].nunique().reset_index().sort_values(by='dataset',ascending=False)
    else:
        best_final = all_survie2.sort_values(by = "HR", ascending=False)[55:21683]
        best_final = best_final[best_final.soustype == soustype]
        # res = res_all.sort_values(by = 'dataset', ascending = False).reset_index()
        res = best_final.groupby(["gene", "conclusion"])["dataset"].nunique().reset_index().sort_values(by='dataset',ascending=False)

    # Pour recupérer toutes les lignes pour (gène, conclusion, type survie et sous type)
    def pos(gene_hr_values):
        positive = all(h > 0 for h in gene_hr_values if h != 0)
        negative = all(h < 0 for h in gene_hr_values if h != 0)
        return positive or negative
    gg = pd.DataFrame()
    moy_HR = list()
    nb_HR = list()
    i = 0
    # while gg.shape[0] < 20:
    while i < res.shape[0]:
        gene = best_final[best_final.gene == res.iloc[i]['gene']][
                best_final.conclusion == res.iloc[i]['conclusion']]
        if pos(gene['HR'].tolist()):
            gg = gg.append(res.iloc[i])
            moy_HR.append(gene['HR'].abs().mean())
            nb_HR.append(len(gene['HR']))
        i = i + 1
    gg['moy_HR'] = moy_HR
    gg['nb_HR'] = nb_HR
    gg = gg[gg.nb_HR > nb]
    gg = gg.sort_values(by='moy_HR',ascending=False)
    gg = gg[:20]
    # gg = pd.concat([res[res.conclusion == "Not Significative"].head(10),
    #        res[res.conclusion != "Not Significative"][res.conclusion != "NaN"].head(10)])
    f = pd.DataFrame()
    for g,ccl in zip(gg.gene,gg.conclusion):
         f = pd.concat([f,best_final[best_final.gene == g][best_final.conclusion == ccl]])
    f = f.reset_index()
    f = f[['dataset', 'gene', 'HR']]
    f_mod = f.assign(abs_HR=best_final['HR'].abs()).sort_values(by=['dataset', 'gene', 'abs_HR'], ascending=[True, True, False])
    f = f_mod.drop_duplicates(subset=['dataset', 'gene']).drop(columns=['abs_HR'])

    # Pour griser les valeurs manquantes
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    cmap.set_bad('lightgrey')

    plt.figure(figsize=(15, 12)) 
    ax = sns.heatmap(f.pivot("dataset", "gene", "HR"), annot=True, cmap=cmap, fmt=".2f",center = 0,square = True,cbar_kws={"shrink": 0.2}, linecolor='white', linewidths=0.5)
    # plt.title(f"Hazard Ratios pour différentes paires de datasets et genes - {soustype}")
    plt.xlabel('')
    plt.ylabel('')
    cbar = ax.collections[0].colorbar
    cbar.set_label('log(Hazard Ratio)', rotation=-270, labelpad=5)
    plt.show()
    return plt

# Test pour All-tumours
plot_HR("All-tumours",2)