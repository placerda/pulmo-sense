# Overview

Esta pagina faz um overview do trabalho e repositório.

## Resultados

## Dataset Utilizado

-  [CC-CCII Dataset](http://ncov-ai.big.ac.cn/download)  
Kang Zhang, Xiaohong Liu, Jun Shen, et al. (2020). *Clinically Applicable AI System for Accurate Diagnosis, Quantitative Measurements and Prognosis of COVID-19 Pneumonia Using Computed Tomography*. Cell. DOI: [10.1016/j.cell.2020.04.045](https://www.cell.com/cell/fulltext/S0092-8674\(20\)30551-1?rss=yes)

**Estado Original do Dataset**

    Common Pneumonia - 964 pacientes, 1347 Volumes, 154802 Slices
    Novel COVID Pneumonia - 897 pacientes, 1449 Volumes, 149036 Slices
    Normal - 849 pacientes, 1046 Volumes, 1046 Slices
    Total - 2710 pacientes, 3842 Volumes, 398777 Slices

## Amostras

## Setup Local

clone https://github.com/placerda/pulmo-sense.git

cd pulmo-sense

python -m venv .venv

.venv\Scripts\Activate.ps1   

## Setup Cloud

## Preparação

**Primeira Limpeza de Dados**

Remocao dos casos com mais de um exame por folder e menos de 30 cortes.

    Common Pneumonia - 580 pacientes, 1124 Volumes 
    Novel COVID Pneumonia - 796 pacientes, 1348 Volumes 
    Normal - 849 pacientes, 1046 Volumes
    Total - 2127 pacientes, 3445 Volumes 

**Segunda Limpeza de dados**

Remoção dos casos de exames segmentados.

    Common Pneumonia - 577 pacientes, 1119 Volumes 
    Novel COVID Pneumonia - 707 pacientes, 1237 Volumes 
    Normal - 151 pacientes, 373 Volumes
    Total - 1435 pacientes, 2729 Volumes 

**Particionamento em test e train**

Train

    Normal - 135 pacientes, 330 Volumes
    CP - 519 pacientes, 1005 Volumes
    NCP - 636 pacientes, 1108 Volumes
    Total - 1290 pacientes, 2443 Volumes

Test

    Normal - 16 pacientes, 43 Volumes
    CP - 58 pacientes, 114 Volumes
    NCP - 71 pacientes, 129 Volumes
    Total - 145 pacientes, 286 Volumes


## Treinamento (scripts, jobs e logs na cloud)

## Modelos

## Teste

## Inferencia