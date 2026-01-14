# scLAMBDA

*Modeling and predicting single-cell multi-gene perturbation responses with scLAMBDA*

We present scLAMBDA, a deep generative learning framework designed to model and predict single-cell transcriptional responses to genetic perturbations, including single-gene and combinatorial multi-gene perturbations. 

By leveraging the embedding capabilities of large language models, scLAMBDA effectively predicts genetic perturbation outcomes for unobserved target genes or gene combinations. Its disentangled representation learning framework enables the modeling of single-cell level perturbations by separating basal cell representations from salient representations associated with perturbation states, allowing for single-cell level generation and *in silico* prediction of perturbation responses. 

![scLAMBDA_overview](https://github.com/gefeiwang/scLAMBDA/blob/main/demos/overview.png)

## Installation
scLAMBDA can be installed from from GitHub:
```bash
git clone https://github.com/gefeiwang/scLAMBDA.git
cd scLAMBDA
conda env update --f environment.yml
conda activate sclambda
```
Normally the installation time is less than 5 minutes.

## Quick Start

### Preprocessing
scLAMBDA accepts logarithmized data stored in an AnnData object as input. The target genes should be specified in `adata.obs['condition']`. For single-gene perturbation datasets, this field should contain entries such as `"gene_a"` for perturbed cells and `"ctrl"` for control cells. For two-gene perturbation datasets, entries should follow the format `"gene_a+gene_b"` for cells with two target genes, `"gene_a+ctrl"` for cells with one target gene, or `"ctrl"` for control cells.

To run scLAMBDA, a dictionary of gene embeddings in the form `gene_embeddings = {"gene_name": np.array}` must also be provided as input.

The function can be used for simulating a dataset splitting:
```python
n_split = 0
# for single-gene perturbation
adata, split = sclambda.utils.data_split(adata, split_type='single', seed=n_split)
# for two-gene perturbation
adata, split = sclambda.utils.data_split(adata, seed=n_split)
# use all available perturbations in training:
adata, split = sclambda.utils.data_split(adata, split_type='all_train', seed=n_split)
```
### Training
To train the scLAMBDA model, use:
```python
model = sclambda.model.Model(adata, 
                             gene_embeddings,
                             model_path = 'path_to_save_dir/models')

model.train()
```
Note that for single-gene perturbations, `multi_gene = False` should be included in the `sclambda.model.Model()` parameters.
* (Optional) By setting `use_tg_coord = True`, the model leverages the indices of perturbation target genes to improve the target-gene effect prediction.

For multi-cell line training, context embeddings encoding cell line or cell type information should be included using the following script:
```python
model = sclambda.model.Model_context(adata, 
                                     gene_embeddings,
                                     ct_embeddings,
                                     model_path = 'path_to_save_dir/models')

model.train()
```
For this utility, three columns should be included in `adata.obs`: the target genes specified in `adata.obs['condition']`, the cell types specified in `adata.obs['cell_type']`, and a combined column `adata.obs['cell_type+condition']`, which can be created as follows:
```python
adata.obs['cell_type+condition'] = [adata.obs['cell_type'].values[i] + '_---_' + adata.obs['condition'].values[i] for i in range(adata.shape[0])]
```
should be included.
### Prediction
Once trained, scLAMBDA can be used for *in silico* predictions. With a list of test perturbations `pert_test`, you can perform *in silico* perturbation using:
```python
res = model.predict(pert_test, return_type = 'cells')
```
which will return the prediction results on control cells used in training. With a NumPy array containing the gene expression matrix of control cells `ctrl_exp`, the *in silico* perturbation can be performed on the input control cells:
```python
res = model.predict(pert_test, ctrl_exp = ctrl_exp, return_type = 'cells')
```
Alternatively, with basal states generated from a standard normal prior, you can generate new perturbed cells with a given sample size `n_cells` using:
```python
res = model.generate(pert_test, n_cells = 10000, return_type = 'cells')
```
Setting `return_type='mean'` in both modes returns the prediction results as the mean gene expression.
## Vignettes
User can follow the [example](https://github.com/gefeiwang/scLAMBDA/blob/main/demos/Norman_tutorial.ipynb) for training and evaluating scLAMBDA on the Perturb-seq dataset from Norman et al. (https://www.science.org/doi/10.1126/science.aax4438). The multi-cell line training code can be found in the [example](https://github.com/gefeiwang/scLAMBDA/blob/main/demos/multi_cell_line_for_jurkat_eval.ipynb), which uses essentail gene perturbations from four cell lines (https://www.sciencedirect.com/science/article/pii/S0092867422005979; https://www.nature.com/articles/s41588-025-02169-3).
## Citation
Gefei Wang, Tianyu Liu, Jia Zhao, Youshu Cheng, Hongyu Zhao. Modeling and predicting single-cell multi-gene perturbation responses with scLAMBDA. bioRxiv 2024; doi: https://doi.org/10.1101/2024.12.04.626878.
