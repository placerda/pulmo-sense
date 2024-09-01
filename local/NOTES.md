
1) Datasets

	- MosMed-1110: COVID-19 and NO-COVID-19
	- COVID-CT-MD: COVID-19, CAP and Normal
	- LUNA16: LUNG CANCER
    - NOVO: CC-CCII: COVID-19 (929), CAP (964) and Normal (849)
            - http://ncov-ai.big.ac.cn/download

2) Data distribution
Samples

| Dataset          |   Samples | Samples by Class                                                    | Percentages by Class                                                         |
|:-----------------|----------:|:--------------------------------------------------------------------|:-----------------------------------------------------------------------------|
| MosMedDataset    |      1110 | COVID-19 Cases: 856, Normal Cases: 254                              | COVID-19 Cases: 77.12%, Normal Cases: 22.88%                                 |
| CovidCtMdDataset |       305 | Cap Cases: 60, COVID-19 Cases: 169, Normal Cases: 76                | Cap Cases: 19.67%, COVID-19 Cases: 55.41%, Normal Cases: 24.92%              |
| Luna16Dataset    |       884 | Nodules: 884                                                        | Nodules: 100.00%                                                             |
| CCCCIIDataset    |      2460 | COVID-19 Cases: 764, Common Pneumonia Cases: 872, Normal Cases: 824 | COVID-19 Cases: 31.06%, Common Pneumonia Cases: 35.45%, Normal Cases: 33.50% |
| CombinedDataset  |      4759 | Abnormal: 3605, Normal: 1154                                        | Abnormal: 75.75%, Normal: 24.25%                                             |

Train / Validation

| Dataset          | Train/Val by Class                                                                  | Percentages by Class                                                                                  |
|:-----------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|
| MosMedDataset    | Train: COVID-19 685 Normal 203 - Val: COVID-19 171 Normal 51                        | Train: COVID-19 77.12% Normal 22.88% - Val: COVID-19 77.12% Normal 22.88%                             |
| CovidCtMdDataset | Train: Cap 48 COVID-19 135 Normal 61 - Val: Cap 12 COVID-19 34 Normal 15            | Train: Cap 19.67% COVID-19 55.41% Normal 24.92% - Val: Cap 19.67% COVID-19 55.41% Normal 24.92%       |
| Luna16Dataset    | Train: Nodules 707 - Val: Nodules 177                                               | Train: Nodules 100.00% - Val: Nodules 100.00%                                                         |
| CCCCIIDataset    | Train: COVID-19 611 Common 698 Normal 659 - Val: COVID-19 153 Common 174 Normal 165 | Train: COVID-19 31.06% Common 35.45% Normal 33.50% - Val: COVID-19 31.06% Common 35.45% Normal 33.50% |
| CombinedDataset  | Train: Normal 923 Abnormal 2884 - Val: Normal 231 Abnormal 721                      | Train: Normal 24.25% Abnormal 75.75% - Val: Normal 24.25% Abnormal 75.75%                             |

**Julho - Setembro** ⏳

Realizar Experimentos comparando diferentes arquiteturas:

- CNN-LSTM ❌
- CNN básica ⬅️
- VGG
- Vision Transformer



3) Novos experimentos 
- **Classificador binário** (COVID-19, NO-COVID-19)
    - Treinar CNN-LSTM com MosMed full. - ok (nao convergiu)
        - Validation Loss: 0.5624, Validation Accuracy: 75.45%, Recall: 1.00, Precision: 0.75, F1 Score: 0.86
    - Testar CNN-LSTM com dataset combinado (10%).
        - Validation Loss: 0.3649, Validation Accuracy: 85.43%, Recall: 1.00, Precision: 0.85, F1 Score: 0.92
    - Fazer análise do dataset novo e ver balanceamento e recall.
        - Dataset desbalanceado
            - Mosmed 77% COVID vs 23 Normal
            - Combined 86% Abnormal 14% Normal
    > - Criar dataset balanceado e retreinar.
        1) Adicionar Mais Casos normais 

    > Testar com dataset CC-CCII balanceado: COVID-19 (929), CAP (964) and Normal (849)
    -> binario(normal/covid) (testar com CNN e CNN-LSTM)
    - tres classes, teste realizado com esse resultado: Validation Loss: 0.8890, Validation Accuracy: 66.87%, Recall: 0.67, Precision: 0.69, F1 Score: 0.67
    - Comparar com CNN basica.
    - Comparar com VGG.
    - Comparar com Vision Transformer.

- **Motivacao para  fazer Normal vs. Doenca**
    - Tendencia das doencas atualmente
    - Baixo desempenho de modelos especialistas quando ha outras doencas.
     



**Agosto**

- **Texto com novos resultados** (Final Agosto)










- **Classificador multi-classe** (COVID-19, CAP, CANCER, FIBROSIS)
    - Treinado com (MosMed Full, COVID-CT-MD 10%, LUNA16 90% e OSIC 90%)
    - Testar com (COVID-CT-MD 10%, LUNA16 10% e OSIC 10%)
    - OBS: revisar distribuição de COVID-CT-MD para train/test
